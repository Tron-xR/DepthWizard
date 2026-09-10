using System;
using UnityEngine;

namespace DepthWizard.Terrain
{
    public class MeshGenerator : MonoBehaviour
    {
        [SerializeField] private MeshFilter _meshFilter;
        [SerializeField] private MeshRenderer _meshRenderer;
        [SerializeField] private Material _terrainMaterial;

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
            float elevRange = Mathf.Max(maxElev - minElev, 0.01f);

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
                    float wy = heights[i];

                    vertices[i] = new Vector3(wx, wy, wz);
                    uvs[i] = new Vector2(nx, ny);

                    float t = (heights[i] - minElev) / elevRange;
                    colors[i] = Color.Lerp(Color.black, Color.white, t);
                }
            }

            Debug.Log(
                $"Terrain corners: " +
                $"(0,0)=({vertices[0].x:F1},{vertices[0].y:F1},{vertices[0].z:F1}) " +
                $"(w,0)=({vertices[width - 1].x:F1},{vertices[width - 1].y:F1},{vertices[width - 1].z:F1}) " +
                $"(0,d)=({vertices[width * (height - 1)].x:F1}," +
                $"{vertices[width * (height - 1)].y:F1},{vertices[width * (height - 1)].z:F1}) " +
                $"(w,d)=({vertices[width * height - 1].x:F1},{vertices[width * height - 1].y:F1}," +
                $"{vertices[width * height - 1].z:F1})");

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
        /// at that point, or -1 if no hit.
        /// </summary>
        public float QueryHeight(Vector3 worldPos)
        {
            if (_mesh == null) return -1f;

            RaycastHit hit;
            Vector3 origin = new Vector3(worldPos.x, 10000f, worldPos.z);
            if (Physics.Raycast(origin, Vector3.down, out hit, 20000f))
                return hit.point.y;
            return -1f;
        }

        /// <summary>
        /// Compute slope (degrees from vertical) at a given world position
        /// by querying the mesh normal via raycast.
        /// </summary>
        public float QuerySlope(Vector3 worldPos)
        {
            if (_mesh == null) return 0f;

            RaycastHit hit;
            Vector3 origin = new Vector3(worldPos.x, 10000f, worldPos.z);
            if (Physics.Raycast(origin, Vector3.down, out hit, 20000f))
            {
                float angle = Vector3.Angle(hit.normal, Vector3.up);
                return angle;
            }
            return 0f;
        }
    }
}
