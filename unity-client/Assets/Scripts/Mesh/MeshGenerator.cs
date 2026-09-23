using System;
using UnityEngine;

namespace DepthWizard.Terrain
{
    public class MeshGenerator : MonoBehaviour
    {
        [SerializeField] private MeshFilter _meshFilter;
        [SerializeField] private MeshRenderer _meshRenderer;
        [SerializeField] private Material _terrainMaterial;

        // When the elevation span is at/near zero (flat / degenerate tile) the
        // black->white height ramp collapses to a solid black surface, which
        // reads as a rendering error rather than an intentional flat result.
        // Render flat meshes in a distinct neutral water-blue instead. The
        // normal Lerp(black, white, t) colouring is untouched for real relief.
        private const float _FLAT_ELEV_SPAN_M = 0.5f;
        private static readonly Color _FlatColor = new Color(0.55f, 0.75f, 0.95f, 1f);

        // Display-only vertical exaggeration, applied to vertex Y AFTER
        // SmoothHeights. 1.0 = true scale. This value never feeds the server,
        // /validate, /evaluate, or any metric path — it only scales the mesh
        // for visibility. The HUD raycast readout divides back by this factor
        // so reported height/slope stay true-scale.
        public float verticalExaggeration = 3.0f;

        private Mesh _mesh;
        private Mesh _wireMesh;
        private bool _wireframeOn;

        public Mesh GeneratedMesh => _mesh;
        public MeshRenderer MeshRenderer => _meshRenderer;
        public bool IsWireframe => _wireframeOn;

        public void ToggleWireframe()
        {
            if (_meshFilter == null) return;
            _wireframeOn = !_wireframeOn;
            if (_wireframeOn && _wireMesh == null && _mesh != null)
                _wireMesh = BuildWireframeMesh();
            _meshFilter.mesh = _wireframeOn ? _wireMesh : _mesh;
        }

        private Mesh BuildWireframeMesh()
        {
            Vector3[] v = _mesh.vertices;
            int[] tri = _mesh.triangles;
            Vector3[] lines = new Vector3[tri.Length];
            for (int i = 0; i < tri.Length; i++)
                lines[i] = v[tri[i]];

            Mesh m = new Mesh { name = "TerrainWire" };
            m.indexFormat = _mesh.indexFormat;
            m.vertices = lines;
            int[] indices = new int[tri.Length];
            for (int i = 0; i < indices.Length; i++) indices[i] = i;
            m.SetIndices(indices, MeshTopology.Lines, 0);
            m.RecalculateBounds();
            return m;
        }

        public void Build(Texture2D heightmap, Texture2D texture, float minElev, float maxElev,
                          float worldWidth, float worldDepth)
        {
            if (_meshFilter == null) _meshFilter = GetComponent<MeshFilter>();
            if (_meshRenderer == null) _meshRenderer = GetComponent<MeshRenderer>();

            int width = heightmap.width;
            int height = heightmap.height;

            float hi = float.MinValue, lo = float.MaxValue, sum = 0f;
            int stride = Mathf.Max(1, width / 32);
            int n = 0;
            for (int y = 0; y < height; y += stride)
                for (int x = 0; x < width; x += stride)
                {
                    float g = heightmap.GetPixel(x, y).grayscale;
                    hi = Mathf.Max(hi, g); lo = Mathf.Min(lo, g); sum += g; n++;
                }
            Debug.Log(
                $"Heightmap grayscale: min {lo:F3} / mean {sum / n:F3} / max {hi:F3}");

            Debug.Log(
                $"MeshGenerator.Build: heightmap {width}x{height}, elev {minElev}..{maxElev}, " +
                $"world {worldWidth}x{worldDepth}");

            float[] heights = SmoothHeights(
                SampleHeightmap(heightmap, minElev, maxElev), width, height, 2);
            LogElevationTrace("heightmap", heightmap, minElev, maxElev);
            LogElevationTrace("heights", heights, minElev, maxElev);
            float elevSpan = maxElev - minElev;
            float elevRange = Mathf.Max(elevSpan, 0.01f);

            _mesh = new Mesh { name = "Terrain" };

            // Unity limits 65535 vertices on old mesh format. Switch to 32-bit
            // index format when the heightmap exceeds that.
            if ((long)width * height > 65535)
                _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

            Vector3[] vertices = new Vector3[width * height];
            Vector2[] uvs = new Vector2[width * height];
            Color[] colors = new Color[width * height];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int i = y * width + x;
                    float nx = (float)x / (width - 1);
                    float ny = (float)y / (height - 1);

                    float wx = nx * worldWidth;
                    float wz = ny * worldDepth;
                    float wy = heights[i] * (Mathf.Abs(verticalExaggeration) < 0.001f ? 1f : verticalExaggeration);

                    vertices[i] = new Vector3(wx, wy, wz);
                    uvs[i] = new Vector2(nx, ny);

                    if (elevSpan <= _FLAT_ELEV_SPAN_M)
                    {
                        // Flat mesh: one tinted colour everywhere instead of a
                        // black ramp that looks like a broken render.
                        colors[i] = _FlatColor;
                    }
                    else
                    {
                        float t = (heights[i] - minElev) / elevRange;
                        colors[i] = Color.Lerp(Color.black, Color.white, t);
                    }
                }
            }

            Debug.Log(
                $"Terrain corners: xv={verticalExaggeration:F2}x " +
                $"(0,0)=({vertices[0].x:F1},{vertices[0].y:F1},{vertices[0].z:F1}) " +
                $"(w,0)=({vertices[width - 1].x:F1},{vertices[width - 1].y:F1},{vertices[width - 1].z:F1}) " +
                $"(0,d)=({vertices[width * (height - 1)].x:F1}," +
                $"{vertices[width * (height - 1)].y:F1},{vertices[width * (height - 1)].z:F1}) " +
                $"(w,d)=({vertices[width * height - 1].x:F1},{vertices[width * height - 1].y:F1}," +
                $"{vertices[width * height - 1].z:F1})");

            float[] vY = new float[vertices.Length];
            for (int i = 0; i < vertices.Length; i++) vY[i] = vertices[i].y;
            LogElevationTrace("vertices.y", vY, minElev, maxElev);

            int[] triangles = new int[(width - 1) * (height - 1) * 6];
            int tri = 0;

            for (int y = 0; y < height - 1; y++)
            {
                for (int x = 0; x < width - 1; x++)
                {
                    int bl = y * width + x;
                    int br = bl + 1;
                    int tl = bl + width;
                    int tr = tl + 1;

                    triangles[tri++] = bl;
                    triangles[tri++] = tl;
                    triangles[tri++] = tr;

                    triangles[tri++] = bl;
                    triangles[tri++] = tr;
                    triangles[tri++] = br;
                }
            }

            _mesh.Clear();
            _mesh.vertices = vertices;
            _mesh.triangles = triangles;
            _mesh.uv = uvs;
            _mesh.colors = colors;
            _mesh.RecalculateNormals();
            _mesh.RecalculateBounds();
            Debug.Log($"Normals: {_mesh.normals.Length}, sample={_mesh.normals[0]}");
            Debug.Log(
                $"[elevtrace] mesh.bounds: min.y={_mesh.bounds.min.y:F3} max.y={_mesh.bounds.max.y:F3} " +
                $"yRange={_mesh.bounds.max.y - _mesh.bounds.min.y:F3} " +
                $"size={_mesh.bounds.size.x:F1}x{_mesh.bounds.size.y:F1}x{_mesh.bounds.size.z:F1} " +
                $"vertexCount={_mesh.vertexCount}");

            _meshFilter.mesh = _mesh;

            MeshCollider collider = GetComponent<MeshCollider>();
            if (collider == null) collider = gameObject.AddComponent<MeshCollider>();
            collider.sharedMesh = _mesh;

            if (_meshRenderer != null)
            {
                Material mat;
                if (_terrainMaterial != null)
                {
                    mat = new Material(_terrainMaterial);
                }
                else
                {
                    // URP shaders render magenta unless URP is the ACTIVE pipeline.
                    Shader shader =
                        UnityEngine.Rendering.GraphicsSettings.defaultRenderPipeline != null
                            ? Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard")
                            : Shader.Find("Standard");
                    shader = shader ?? Shader.Find("Sprites/Default");
                    mat = new Material(shader);
                }
                if (texture != null)
                {
                    mat.SetTexture("_BaseMap", texture);
                    mat.mainTexture = texture;
                }
                _meshRenderer.material = mat;
                Debug.Log(
                    $"Terrain material: {mat.shader.name}; texture: " +
                    (texture != null ? $"{texture.width}x{texture.height}" : "NONE"));
            }
        }

        /// <summary>
        /// Diagnostic trace of the heightmap->mesh elevation path (temporary).
        /// </summary>
        private static void LogElevationTrace(string label, Texture2D tex, float minElev, float maxElev)
        {
            int w = tex.width, h = tex.height;
            float[] g = new float[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    g[y * w + x] = tex.GetPixel(x, y).grayscale;
            Debug.Log($"[elevtrace] {label}: {w}x{h} format={tex.format}");
            LogElevationTrace($"{label}.grayscale", g, 0f, 1f);
            float[] e = new float[g.Length];
            for (int i = 0; i < g.Length; i++) e[i] = Mathf.Lerp(minElev, maxElev, g[i]);
            LogElevationTrace($"{label}.elevation", e, minElev, maxElev);
        }

        private static void LogElevationTrace(string label, float[] values, float minElev, float maxElev)
        {
            int n = values.Length;
            float sum = 0f, min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                float v = values[i];
                sum += v;
                if (v < min) min = v;
                if (v > max) max = v;
            }
            float mean = sum / n;
            float ssum = 0f;
            for (int i = 0; i < n; i++) { float d = values[i] - mean; ssum += d * d; }
            float std = Mathf.Sqrt(ssum / n);

            float[] sorted = (float[])values.Clone();
            Array.Sort(sorted);
            Debug.Log(
                $"[elevtrace] {label}: n={n} min={min:F3} max={max:F3} range={max - min:F3} " +
                $"mean={mean:F3} std={std:F3} " +
                $"p1={sorted[Mathf.Max(0, (int)(n * 0.01f))]:F3} " +
                $"p5={sorted[Mathf.Max(0, (int)(n * 0.05f))]:F3} " +
                $"p50={sorted[(int)(n * 0.50f)]:F3} " +
                $"p95={sorted[Mathf.Min(n - 1, (int)(n * 0.95f))]:F3} " +
                $"p99={sorted[Mathf.Min(n - 1, (int)(n * 0.99f))]:F3} " +
                $"elevMin={minElev:F3} elevMax={maxElev:F3} elevRange={maxElev - minElev:F3}");
        }

        /// <summary>
        /// Box-blur world heights to suppress small-scale depth-model ripple
        /// which otherwise reads as corrugation once lit in 3D.
        /// </summary>
        private static float[] SmoothHeights(float[] h, int width, int height, int passes)
        {
            for (int p = 0; p < passes; p++)
            {
                float[] next = new float[h.Length];
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                    {
                        float sum = 0f;
                        int n = 0;
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                int nx = x + dx, ny = y + dy;
                                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                                sum += h[ny * width + nx];
                                n++;
                            }
                        next[y * width + x] = sum / n;
                    }
                h = next;
            }
            return h;
        }

        /// <summary>
        /// Sample heightmap into a flat array of world-space Y values.
        /// Expects a grayscale PNG where black = minElev, white = maxElev.
        /// </summary>
        private float[] SampleHeightmap(Texture2D tex, float minElev, float maxElev)
        {
            int w = tex.width;
            int h = tex.height;
            float[] out_ = new float[w * h];

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    float t = tex.GetPixel(x, y).grayscale;
                    out_[y * w + x] = Mathf.Lerp(minElev, maxElev, t);
                }
            }
            return out_;
        }

        /// <summary>
        /// Raycast from a world position downward; returns the terrain height
        /// at that point (true-scale, exaggeration divided back out), or -1 if
        /// no hit.
        /// </summary>
        public float QueryHeight(Vector3 worldPos)
        {
            if (_mesh == null) return -1f;

            RaycastHit hit;
            Vector3 origin = new Vector3(worldPos.x, 10000f, worldPos.z);
            if (Physics.Raycast(origin, Vector3.down, out hit, 20000f))
                return hit.point.y / _TrueScaleFactor();
            return -1f;
        }

        /// <summary>
        /// Compute slope (degrees from vertical) at a given world position.
        /// The mesh is vertically exaggerated by verticalExaggeration, which
        /// scales tan(slope) by that factor; invert it so the result is the
        /// true slope of the unexaggerated terrain.
        /// </summary>
        public float QuerySlope(Vector3 worldPos)
        {
            if (_mesh == null) return 0f;

            RaycastHit hit;
            Vector3 origin = new Vector3(worldPos.x, 10000f, worldPos.z);
            if (Physics.Raycast(origin, Vector3.down, out hit, 20000f))
            {
                float ex = Vector3.Angle(hit.normal, Vector3.up) * Mathf.Deg2Rad;
                float k = _TrueScaleFactor();
                float trueRad = Mathf.Atan(Mathf.Tan(ex) / k);
                return trueRad * Mathf.Rad2Deg;
            }
            return 0f;
        }

        /// <summary>
        /// Effective scale divisor: exaggeration clamped so it can never be 0
        /// or negative (which would flip/zero heights).
        /// </summary>
        private float _TrueScaleFactor()
        {
            return Mathf.Abs(verticalExaggeration) < 0.001f ? 1f : verticalExaggeration;
        }
    }
}
