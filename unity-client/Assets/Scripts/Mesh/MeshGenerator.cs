using System;
using System.Collections.Generic;
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
        // /validate, /evaluate, or any metric path  it only scales the mesh
        // for visibility. The HUD raycast readout divides back by this factor
        // so reported height/slope stay true-scale.
        public float verticalExaggeration = 3.0f;

        // Solid-block extrusion: when true the top relief is extruded downward
        // into a block (top surface + vertical side walls + flat bottom cap),
        // matching the "lifted-out chunk of terrain" look instead of a floating
        // heightmap sheet. Purely presentational  widths/heights/slopes are
        // computed identically either way.
        public bool extrudeAsBlock = true;

        // Bottom cap depth as a fraction of the mesh's world height range,
        // measured below the lowest top vertex. Exposed in the Inspector.
        [Range(0.02f, 0.5f)] public float baseDepth = 0.15f;

        // Optional flat material for the vertical walls + bottom cap. When null
        // a default dark-brown material is created so the block reads as solid
        // earth rather than a textured underside.
        public Material sideMaterial;

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
            Debug.Log($"Heightmap grayscale: min {lo:F3} / mean {sum / n:F3} / max {hi:F3}");

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
                    float wy = heights[i] *
                        (Mathf.Abs(verticalExaggeration) < 0.001f ? 1f : verticalExaggeration);

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

            if (extrudeAsBlock)
            {
                ExtrudeToBlock(vertices, uvs, colors, triangles, width, height, worldWidth, worldDepth);
            }
            else
            {
                _mesh.Clear();
                _mesh.vertices = vertices;
                _mesh.triangles = triangles;
                _mesh.uv = uvs;
                _mesh.colors = colors;
                _mesh.RecalculateNormals();
                _mesh.RecalculateBounds();
            }

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

                if (extrudeAsBlock)
                {
                    Shader sideShader =
                        UnityEngine.Rendering.GraphicsSettings.defaultRenderPipeline != null
                            ? Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard")
                            : Shader.Find("Standard");
                    sideShader = sideShader ?? Shader.Find("Sprites/Default");
                    Material sideMat = sideMaterial != null ? new Material(sideMaterial) : new Material(sideShader);
                    if (sideMaterial == null) sideMat.color = _WallColor;
                    _meshRenderer.materials = new[] { mat, sideMat };
                }
                else
                {
                    _meshRenderer.material = mat;
                }
                Debug.Log(
                    $"Terrain material: {mat.shader.name}; texture: " +
                    (texture != null ? $"{texture.width}x{texture.height}" : "NONE"));
            }
        }

        /// <summary>
        /// Solid-wall colour for the extruded block (dark earth-brown, RGB
        /// ~0.20/0.18/0.15). Used only for wall + bottom-cap vertices when
        /// extrudeAsBlock is on; the top relief keeps its heightmap ramp.
        /// </summary>
        private static readonly Color _WallColor = new Color(0.20f, 0.18f, 0.15f, 1f);

        private const float _EPSILON_Y = 0.001f;

        /// <summary>
        /// Extrude the flat top-surface (vertices/triangles) down into a solid
        /// block: a single mesh with TWO submeshes  submesh 0 = the original
        /// top surface, submesh 1 = vertical side walls + flat bottom cap.
        /// The winding of each wall/cap triangle is oriented so the face normal
        /// points outward (walls) or down (cap). If a wall/cap renders culled
        /// hidden in-editor, flip the vertex order of that quad's two triangles
        /// (see note 2d).
        /// </summary>
        private void ExtrudeToBlock(Vector3[] topVerts, Vector2[] topUvs, Color[] topColors,
                                    int[] topTris, int width, int height,
                                    float worldWidth, float worldDepth)
        {
            float minTopY = float.MaxValue, maxTopY = float.MinValue;
            for (int i = 0; i < topVerts.Length; i++)
            {
                minTopY = Mathf.Min(minTopY, topVerts[i].y);
                maxTopY = Mathf.Max(maxTopY, topVerts[i].y);
            }
            float yRange = maxTopY - minTopY;
            float baseY = minTopY - yRange * baseDepth;
            if (yRange < _EPSILON_Y) baseY = minTopY - 1f;

            List<Vector3> verts = new List<Vector3>(topVerts);
            List<Vector2> uvs = new List<Vector2>(topUvs);
            List<Color> cols = new List<Color>(topColors);

            // Perimeter loop over the four edges (counter-clockwise when viewed
            // from above), each top vertex gets one duplicate at baseY.
            List<int> perim = new List<int>();
            for (int x = 0; x < width; x++) perim.Add(x);
            for (int y = 1; y < height; y++) perim.Add(y * width + (width - 1));
            for (int x = width - 2; x >= 0; x--) perim.Add((height - 1) * width + x);
            for (int y = height - 2; y >= 1; y--) perim.Add(y * width);

            int topCount = topVerts.Length;
            int capIdx = topCount + perim.Count; // single centre vertex for the cap fan

            Vector3 capCenter = new Vector3(worldWidth * 0.5f, baseY, worldDepth * 0.5f);

            for (int i = 0; i < perim.Count; i++)
            {
                int ti = perim[i];
                Vector3 top = topVerts[ti];
                verts.Add(new Vector3(top.x, baseY, top.z));
                uvs.Add(topUvs[ti]);
                cols.Add(_WallColor);
            }
            verts.Add(capCenter);
            uvs.Add(new Vector2(0.5f, 0.5f));
            cols.Add(_WallColor);

            List<int> sideTris = new List<int>();

            // Wall quads: connect each adjacent top perimeter pair to its base
            // counterpart. Orient winding so the wall normal points outward.
            for (int i = 0; i < perim.Count; i++)
            {
                int j = (i + 1) % perim.Count;
                int aTop = perim[i];
                int bTop = perim[j];
                int aBase = topCount + i;
                int bBase = topCount + j;

                Vector3 outward =
                    new Vector3(
                        (topVerts[aTop].x + topVerts[bTop].x) * 0.5f - worldWidth * 0.5f,
                        0f,
                        (topVerts[aTop].z + topVerts[bTop].z) * 0.5f - worldDepth * 0.5f);
                if (outward.sqrMagnitude < 1e-6f) outward = Vector3.forward;

                EmitOrientedQuad(
                    sideTris, verts, aTop, bTop, bBase, aBase, outward);

                // Bottom cap quadrant for this wall segment, oriented downward.
                EmitOrientedTri(
                    sideTris, verts, capIdx, aBase, bBase, Vector3.down);
            }

            if ((long)verts.Count > 65535)
                _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

            _mesh.Clear();
            _mesh.vertices = verts.ToArray();
            _mesh.uv = uvs.ToArray();
            _mesh.colors = cols.ToArray();
            _mesh.subMeshCount = 2;
            _mesh.SetTriangles(topTris, 0);
            _mesh.SetTriangles(sideTris.ToArray(), 1);
            _mesh.RecalculateNormals();   // AFTER all verts + all tris
            _mesh.RecalculateBounds();    // AFTER all verts + all tris
        }

        /// <summary>
        /// Emit two triangles for a quad (tld, trd, brd, bld) whose winding
        /// makes the face normal point roughly along 'outDir'.
        /// </summary>
        private static void EmitOrientedQuad(List<int> tris, List<Vector3> v,
            int tl, int tr, int bl, int br, Vector3 outDir)
        {
            EmitOrientedTri(tris, v, tl, bl, tr, outDir);
            EmitOrientedTri(tris, v, bl, br, tr, outDir);
        }

        /// <summary>
        /// Emit a single triangle oriented so its geometric normal has a
        /// positive dot with 'desired'. Deterministic: picks the winding.
        /// </summary>
        private static void EmitOrientedTri(List<int> tris, List<Vector3> v,
            int a, int b, int c, Vector3 desired)
        {
            Vector3 n = Vector3.Cross(v[b] - v[a], v[c] - v[a]);
            if (Vector3.Dot(n, desired) >= 0f)
            {
                tris.Add(a); tris.Add(b); tris.Add(c);
            }
            else
            {
                tris.Add(a); tris.Add(c); tris.Add(b);
            }
        }

        /// <summary>
        /// Trace of the heightmap->mesh elevation path (diagnostic, temporary).
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

        private static float[] SampleHeightmap(Texture2D tex, float minElev, float maxElev)
        {
            int w = tex.width, h = tex.height;
            float[] out_ = new float[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float t = tex.GetPixel(x, y).grayscale;
                    out_[y * w + x] = Mathf.Lerp(minElev, maxElev, t);
                }
            return out_;
        }

        private static float[] SmoothHeights(float[] h, int width, int height, int passes)
        {
            for (int p = 0; p < passes; p++)
            {
                float[] next = new float[h.Length];
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                    {
                        float sum = 0f;
                        int nbr = 0;
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                int nx = x + dx, ny = y + dy;
                                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                                sum += h[ny * width + nx];
                                nbr++;
                            }
                        next[y * width + x] = sum / nbr;
                    }
                h = next;
            }
            return h;
        }

        public float QueryHeight(Vector3 worldPos)
        {
            if (_mesh == null) return -1f-1f;
            Vector3 origin = new Vector3(worldPos.x, 10000f, worldPos.z);
            RaycastHit hit;
            // Top surface is hit first; walls/cap never interfere with the
            // downward height query.
            if (Physics.Raycast(origin, Vector3.down, out hit, 20000f))
                return hit.point.y / verticalExaggeration;
            return -1f;
        }

        public float QuerySlope(Vector3 worldPos)
        {
            if (_mesh == null) return 0f;
            Vector3 origin = new Vector3(worldPos.x, 10000f, worldPos.z);
            RaycastHit hit;
            if (Physics.Raycast(origin, Vector3.down, out hit, 20000f))
            {
                float rad = Mathf.Acos(Mathf.Clamp(Vector3.Dot(hit.normal, Vector3.up), -1f, 1f));
                return rad * Mathf.Rad2Deg;
            }
            return 0f;
        }
    }
}
