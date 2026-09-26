using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using DepthWizard.Camera;
using DepthWizard.Core;
using DepthWizard.Terrain;

namespace DepthWizard.UI
{
    public class ViewerScreen : MonoBehaviour
    {
        [Header("UI Elements")]
        [SerializeField] private TextMeshProUGUI _heightText;
        [SerializeField] private TextMeshProUGUI _slopeText;
        [SerializeField] private Button _validateButton;
        [SerializeField] private Button _wireframeButton;
        [SerializeField] private Button _screenshotButton;
        [SerializeField] private Button _menuButton;
        [SerializeField] private TextMeshProUGUI _validationResultText;

        [Header("DEM view overlay")]
        [Tooltip("RawImage that shows the grayscale DEM overlay of the current job. " +
                 "Leave null to skip the feature; ToggleDemView logs an error.")]
        [SerializeField] private RawImage _demViewImage;
        [Tooltip("Optional wrapper panel that shows/hides with the overlay. May be null.")]
        [SerializeField] private GameObject _demViewPanel;

        [Header("Display mode")]
        [Tooltip("Rotation speed of the terrain while in display/showcase mode, degrees per second.")]
        [SerializeField] private float _displayRotationSpeed = 15f;

        [Header("Terrain")]
        [SerializeField] private MeshGenerator _meshGenerator;
        [SerializeField] private float _raycastMaxDist = 10000f;

        [Header("Hit marker")]
        [Tooltip("Flat disc shown at the screen-centre hit point of the same raycast that " +
                 "drives the Height/Slope readout. Size is a fraction of the terrain's bounding size.")]
        [SerializeField] private float _markerSizeFraction = 0.006f;
        [Tooltip("Disc thickness as a fraction of the marker diameter, so it reads as a thin " +
                 "plate lying on the surface rather than a ball.")]
        [SerializeField] private float _markerThicknessFraction = 0.06f;
        [Tooltip("Exponential smoothing factor for the marker's glide toward the raycast hit point; " +
                 "higher = snappier, lower = more lag/glide. Frame-rate independent.")]
        [SerializeField] private float _followSpeed = 13f;
        [Tooltip("Bright, unlit colour for the hit-marker ring so it stands out against the " +
                 "terrain texture and raking-light shadows.")]
        [SerializeField] private Color _markerColor = new Color(0.2f, 0.5f, 1f, 1f);
        /// <summary>Draw a thin ray line from the camera to the hit point. Off by default.</summary>
        public bool showHitRay;

        private GameObject _hitMarker;
        private LineRenderer _hitRayLine;
        private Vector3 _markerTargetPoint;
        private Vector3 _markerTargetNormal;
        private bool _markerHasTarget;
        private bool _markerWasHidden = true;

        [Header("Validation warning")]
        [Tooltip("Colour used to make a low-relief/degenerate validation warning stand out.")]
        [SerializeField] private Color _warningColor = new Color(1f, 0.8f, 0.2f, 1f);

        [Header("Vertical exaggeration")]
        [Tooltip("Display-only vertical exaggeration, default 3x for the georeferenced branch. " +
                 "Applied only in MeshGenerator.Build; never sent to the server or used by metrics.")]
        [SerializeField] private float _exaggerationValue = 3f;

        private Launcher _launcher;
        private ResultResponse _resultData;
        private string _jobId;
        private UnityEngine.Camera _cam;

        private Texture2D _cachedHeightTex;
        private Texture2D _cachedSourceTex;
        private Texture2D _cachedDemTex;
        private bool _demViewVisible;

        // Display (showcase) mode state: while active, the Viewer UI is hidden
        // and the Terrain rotates in place about its own centre on the Y axis.
        // Rotation about world-Y leaves every vertex's world-Y unchanged and
        // every normal's angle to Vector3.up unchanged, so the HUD raycast
        // height/slope math stays exactly correct during and after rotation.
        private bool _displayModeActive;
        private int _displayEnterFrame = -1;
        private readonly List<UiSnapshot> _displayUiSnapshot = new List<UiSnapshot>();

        private struct UiSnapshot
        {
            public GameObject Target;
            public bool Active;
            public bool Interactable;
        }

        // Monotonic token: every new result invalidates any in-flight download
        // from the previous result, so a slow A can never overwrite a newer B.
        private int _renderToken;

        // ProcessingScreen calls SetResultData before ShowViewer, so a new
        // result arrives while this screen is still inactive. Rebuild then
        // (instead of gating on isActiveAndEnabled) so the second upload
        // actually renders instead of keeping the first job's mesh on screen.
        private bool _pendingBuild;

        public void SetResultData(ResultResponse data)
        {
            // New terrain supersedes any prior render: bump the token (drops
            // stale downloads) and clear cached textures so a previous result
            // can never leak into the next terrain. State is per-result, not
            // persistent across uploads.
            _renderToken++;
            _cachedHeightTex = null;
            _cachedSourceTex = null;
            _cachedDemTex = null;
            SetDemViewShown(false);
            _resultData = data;
            if (isActiveAndEnabled)
                StartCoroutine(BuildTerrain());
            else
                _pendingBuild = true;
        }

        public void SetJobId(string jobId) => _jobId = jobId;

        private void Awake()
        {
            _launcher = Object.FindFirstObjectByType<Launcher>(FindObjectsInactive.Include);
            _cam = UnityEngine.Camera.main;
            CreateHitMarker();
        }

        private void OnEnable()
        {
            _validateButton.onClick.AddListener(OnValidate);
            _wireframeButton.onClick.AddListener(OnToggleWireframe);
            _screenshotButton.onClick.AddListener(OnScreenshot);
            _menuButton.onClick.AddListener(OnMenu);
            _validationResultText.text = "";
            _validationResultText.color = Color.white;

            if (_resultData != null && !_resultData.is_georeferenced)
            {
                _validateButton.interactable = false;
                _validationResultText.text = "Validation needs a georeferenced image (GPS/EXIF tags).";
            }

            if (_pendingBuild)
            {
                _pendingBuild = false;
                if (_resultData != null)
                    StartCoroutine(BuildTerrain());
            }
        }

        private void OnDisable()
        {
            // Leaving the viewer (e.g. to the main menu) must never leave the
            // screen half-hidden or the Terrain rotating underneath it.
            if (_displayModeActive) ExitDisplayMode();
            _validateButton.onClick.RemoveListener(OnValidate);
            _wireframeButton.onClick.RemoveListener(OnToggleWireframe);
            _screenshotButton.onClick.RemoveListener(OnScreenshot);
            _menuButton.onClick.RemoveListener(OnMenu);
        }

        private IEnumerator BuildTerrain()
        {
            // Download the heightmap texture
            yield return StartCoroutine(_launcher.Api.DownloadTexture(_resultData.heightmap_url,
                onSuccess: heightTex =>
                {
                    _cachedHeightTex = heightTex;
                    // Download the source texture
                    StartCoroutine(_launcher.Api.DownloadTexture(_resultData.texture_url,
                        onSuccess: sourceTex =>
                        {
                            _cachedSourceTex = sourceTex;
                            BuildMesh();
                        },
                        onError: err =>
                        {
                            Debug.LogWarning($"Texture download error: {err}");
                            _cachedSourceTex = null;
                            BuildMesh();
                        }));
                },
                onError: err =>
                {
                    Debug.LogError($"Heightmap download error: {err}");
                }));
        }

        /// <summary>
        /// Effective vertical exaggeration: georeferenced/absolute branch uses
        /// the Inspector-configured value; the relative/rDSM branch already
        /// spans a fixed 0..200 m over a 1000 m world (about 20% relief) so it
        /// stays at true scale (1x).
        /// </summary>
        private float EffectiveExaggeration()
        {
            if (_resultData != null && !_resultData.is_georeferenced) return 1f;
            return _exaggerationValue;
        }

        private void BuildMesh()
        {
            if (_meshGenerator == null || _cachedHeightTex == null) return;
            _meshGenerator.verticalExaggeration = EffectiveExaggeration();
            _meshGenerator.Build(
                _cachedHeightTex, _cachedSourceTex,
                _resultData.min_elev, _resultData.max_elev,
                _resultData.world_width, _resultData.world_depth);
            FrameTerrain();
        }

        private void FrameTerrain()
        {
            if (_meshGenerator == null || _cam == null || _meshGenerator.MeshRenderer == null) return;

            Bounds b = _meshGenerator.MeshRenderer.bounds;
            Vector3 center = b.center;
            Debug.Log($"Terrain bounds: center {center.ToString("F1")} size {b.size.ToString("F1")}");

            _cam.farClipPlane = Mathf.Max(_cam.farClipPlane, b.size.magnitude * 3f);
            _cam.clearFlags = CameraClearFlags.SolidColor;

            float dist = b.size.magnitude * 1.4f;
            _cam.transform.position = center + new Vector3(0f, dist * 0.42f, -dist * 0.6f);
            _cam.transform.rotation = Quaternion.LookRotation(center - _cam.transform.position, Vector3.up);
            Debug.Log(
                $"Camera after frame: pos {_cam.transform.position.ToString("F1")} " +
                $"rot {_cam.transform.rotation.eulerAngles.ToString("F1")}");

            FreeFlyCamera fly = _cam.GetComponent<FreeFlyCamera>();
            if (fly != null) fly.SyncState();
        }

        private void Update()
        {
            if (_displayModeActive)
            {
                RotateDisplayModel();
                // Skip the click that entered the mode (same frame as the
                // button press); any later mouse/tap click anywhere exits.
                if (Time.frameCount > _displayEnterFrame && ShouldExitDisplay())
                {
                    ExitDisplayMode();
                    return;
                }
            }
            UpdateHUD();
            UpdateHitMarker();
        }

        private void UpdateHUD()
        {
            if (_cam == null) return;
            Ray ray = _cam.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0f));
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit, _raycastMaxDist))
            {
                float k = _meshGenerator != null
                    ? Mathf.Abs(_meshGenerator.verticalExaggeration) < 0.001f
                        ? 1f
                        : _meshGenerator.verticalExaggeration
                    : 1f;
                // Mesh Y is exaggerated for display; divide back so the HUD
                // reports true real-world height, never the visual scale.
                float trueHeight = hit.point.y / k;
                float exSlope = Vector3.Angle(hit.normal, Vector3.up);
                // Vertical scale multiplies tan(slope); invert for true slope.
                float trueSlope = Mathf.Atan(Mathf.Tan(exSlope * Mathf.Deg2Rad) / k) * Mathf.Rad2Deg;
                _heightText.text = $"Height: {trueHeight:F1} m";
                _slopeText.text = $"Slope: {trueSlope:F1}\u00B0";
                _markerTargetPoint = hit.point;
                _markerTargetNormal = hit.normal;
                _markerHasTarget = true;
            }
            else
            {
                _heightText.text = "Height: --";
                _slopeText.text = "Slope: --";
                _markerHasTarget = false;
                HideHitMarker();
            }
        }

        /// <summary>Ring (open annulus) marker at the HUD raycast's hit point so the
        /// Height/Slope readout has a visible spatial anchor on the mesh: a
        /// location reticle rather than a solid dot. Mesh is built at runtime
        /// (no asset) in the XZ plane facing +Y, matching the cylinder axis the
        /// normal-alignment assumes. Created once in Awake, hidden and glued to
        /// the latest hit point by UpdateHitMarker; deliberately unlit and
        /// collider-free so it never feeds the raycast that drives it.</summary>
        private void CreateHitMarker()
        {
            Shader shader = Shader.Find("Unlit/Color") ?? Shader.Find("Sprites/Default");
            Material ringMat = new Material(shader) { color = _markerColor };

            GameObject marker = new GameObject("HitPointMarker");
            marker.AddComponent<MeshFilter>().sharedMesh = BuildRingMesh(0.4f, 0.5f, 64);
            marker.AddComponent<MeshRenderer>().sharedMaterial = ringMat;
            _hitMarker = marker;
            _hitMarker.SetActive(false);

            GameObject rayLine = new GameObject("HitRayLine");
            _hitRayLine = rayLine.AddComponent<LineRenderer>();
            _hitRayLine.positionCount = 2;
            _hitRayLine.startWidth = _hitRayLine.endWidth = 0.02f;
            _hitRayLine.material = new Material(shader) { color = _markerColor };
            rayLine.SetActive(false);
        }

        /// <summary>Thin annulus mesh: a triangle strip between an inner and outer
        /// radius, laid flat in the XZ plane with +Y normals. Outer radius 0.5
        /// matches the built-in cylinder primitive the old disc used, so replacing
        /// this mesh keeps the marker's world size identical for the same
        /// localScale from UpdateHitMarker.</summary>
        private Mesh BuildRingMesh(float innerRadius, float outerRadius, int segments)
        {
            int seg = Mathf.Max(8, segments);
            Vector3[] verts = new Vector3[seg * 2];
            Vector3[] normals = new Vector3[seg * 2];
            for (int i = 0; i < seg; i++)
            {
                float a = i / (float)seg * Mathf.PI * 2f;
                float ca = Mathf.Cos(a), sa = Mathf.Sin(a);
                verts[i * 2] = new Vector3(innerRadius * ca, 0f, innerRadius * sa);
                verts[i * 2 + 1] = new Vector3(outerRadius * ca, 0f, outerRadius * sa);
                normals[i * 2] = normals[i * 2 + 1] = Vector3.up;
            }
            List<int> tris = new List<int>(seg * 6);
            for (int i = 0; i < seg; i++)
            {
                int next = (i + 1) % seg;
                int inA = i * 2, outA = i * 2 + 1;
                int inB = next * 2, outB = next * 2 + 1;
                tris.Add(outA); tris.Add(inA); tris.Add(inB);
                tris.Add(outA); tris.Add(inB); tris.Add(outB);
            }
            Mesh mesh = new Mesh { vertices = verts, triangles = tris.ToArray(), normals = normals };
            mesh.RecalculateBounds();
            return mesh;
        }

        /// <summary>Called every frame from Update. Glides the marker disc toward
        /// the latest raycast hit point and slerps its tilt to the surface normal,
        /// tying both to the same exponential followSpeed so position and alignment
        /// move as one. Hides stay hidden and the next reappearance snaps straight
        /// to the hit point (never glides back in from a stale, far-away position).</summary>
        private void UpdateHitMarker()
        {
            if (_hitMarker == null || !_markerHasTarget) return;

            float scale = 1f;
            if (_meshGenerator != null && _meshGenerator.MeshRenderer != null)
                scale *= _meshGenerator.MeshRenderer.bounds.size.magnitude * _markerSizeFraction;
            float thickness = scale * _markerThicknessFraction;
            Vector3 discScale = new Vector3(scale, thickness, scale);

            // Disc lies in the cylinder's XY plane with +Y as its face normal;
            // rotate so +Y lines up with the terrain normal.
            Quaternion targetRot = Quaternion.FromToRotation(Vector3.up, _markerTargetNormal);
            // Sit the disc face just above the surface: half the thickness plus a
            // small epsilon so it reads as resting on the terrain, not floating.
            Vector3 targetPos = _markerTargetPoint + _markerTargetNormal * (thickness * 0.5f + 0.02f);

            if (_markerWasHidden)
            {
                _hitMarker.transform.localScale = discScale;
                _hitMarker.transform.position = targetPos;
                _hitMarker.transform.rotation = targetRot;
                _hitMarker.SetActive(true);
                _markerWasHidden = false;
            }
            else
            {
                float t = 1f - Mathf.Exp(-_followSpeed * Time.deltaTime);
                _hitMarker.transform.position =
                    Vector3.Lerp(_hitMarker.transform.position, targetPos, t);
                _hitMarker.transform.rotation =
                    Quaternion.Slerp(_hitMarker.transform.rotation, targetRot, t);
                _hitMarker.transform.localScale =
                    Vector3.Lerp(_hitMarker.transform.localScale, discScale, t);
            }

            if (_hitRayLine != null && showHitRay)
            {
                _hitRayLine.SetPosition(0, _cam != null ? _cam.transform.position : _markerTargetPoint);
                _hitRayLine.SetPosition(1, targetPos);
                _hitRayLine.gameObject.SetActive(true);
            }
            else if (_hitRayLine != null)
            {
                _hitRayLine.gameObject.SetActive(false);
            }
        }

        private void HideHitMarker()
        {
            _markerWasHidden = true;
            if (_hitMarker != null) _hitMarker.SetActive(false);
            if (_hitRayLine != null) _hitRayLine.gameObject.SetActive(false);
        }

        private void OnValidate()
        {
            if (string.IsNullOrEmpty(_jobId)) return;

            if (_launcher == null)
                _launcher = Object.FindFirstObjectByType<Launcher>(FindObjectsInactive.Include);
            if (_launcher == null || _launcher.Api == null)
            {
                _validationResultText.text = "Validation unavailable: launcher not found.";
                return;
            }

            _validationResultText.text = "Validating...";
            _validationResultText.color = Color.white;

            StartCoroutine(_launcher.Api.Validate(_jobId,
                onSuccess: resp =>
                {
                    if (resp.degenerate_calibration)
                    {
                        // Flat/degenerate tile: surface the server's flag loudly
                        // instead of presenting RMSE/MAE/correlation as a normal
                        // high-confidence result (they are meaningless on a flat).
                        string reason = string.IsNullOrEmpty(resp.calibration_reason)
                            ? "calibration scale near zero"
                            : resp.calibration_reason;
                        _validationResultText.color = _warningColor;
                        _validationResultText.text =
                            $"Low relief detected \u2014 {reason}\n" +
                            $"RMSE: {resp.rmse:F2} m\n" +
                            $"MAE: {resp.mae:F2} m\n" +
                            $"Correlation: {resp.correlation:F3} (not meaningful for flat terrain)";
                    }
                    else
                    {
                        _validationResultText.color = Color.white;
                        _validationResultText.text =
                            $"RMSE: {resp.rmse:F2} m\n" +
                            $"MAE: {resp.mae:F2} m\n" +
                            $"Correlation: {resp.correlation:F3}\n" +
                            $"Reference: {resp.reference_source}";
                    }
                },
                onError: err =>
                {
                    _validationResultText.color = Color.white;
                    _validationResultText.text = $"Validation error: {err}";
                }));
        }

        /// <summary>
        /// Toggle the grayscale DEM overlay of the current job.
        /// First call: fetch /dem-view/{job_id} (cached per job) and show it.
        /// Second call: hide the overlay without refetching.
        /// Assigned in the Inspector: _demViewImage (RawImage) and the optional
        /// _demViewPanel (GameObject) it sits on.
        /// </summary>
        public void ToggleDemView()
        {
            if (_demViewImage == null)
            {
                Debug.LogError(
                    "ToggleDemView: _demViewImage is not assigned. Drag a RawImage " +
                    "onto ViewerScreen._demViewImage in the Inspector.");
                return;
            }
            if (_launcher == null)
                _launcher = Object.FindFirstObjectByType<Launcher>(FindObjectsInactive.Include);

            if (_demViewVisible)
            {
                SetDemViewShown(false);
                return;
            }
            if (_cachedDemTex != null)
            {
                _demViewImage.texture = _cachedDemTex;
                SetDemViewShown(true);
                return;
            }
            if (string.IsNullOrEmpty(_jobId))
            {
                Debug.LogError("ToggleDemView: no job loaded yet.");
                return;
            }

            StartCoroutine(_launcher.Api.DownloadTexture($"/dem-view/{_jobId}",
                onSuccess: tex =>
                {
                    _cachedDemTex = tex;
                    _demViewImage.texture = tex;
                    SetDemViewShown(true);
                },
                onError: err => Debug.LogError($"DEM view download error: {err}")));
        }

        private void SetDemViewShown(bool shown)
        {
            _demViewVisible = shown;
            if (_demViewImage != null)
                _demViewImage.gameObject.SetActive(shown);
            if (_demViewPanel != null)
                _demViewPanel.SetActive(shown);
        }

        /// <summary>
        /// Toggle display/showcase mode: hides the Viewer UI and spins the
        /// Terrain in place about its centre. Any tap/click exits and restores
        /// every UI element to its exact prior active/interactable state.
        /// </summary>
        public void ToggleDisplayMode()
        {
            if (_displayModeActive) ExitDisplayMode();
            else EnterDisplayMode();
        }

        private void EnterDisplayMode()
        {
            _displayModeActive = true;
            _displayEnterFrame = Time.frameCount;
            CaptureDisplayUi();
            foreach (UiSnapshot s in _displayUiSnapshot)
                s.Target.SetActive(false);
            Debug.Log("Display mode ON: UI hidden, model rotating.");
        }

        private void ExitDisplayMode()
        {
            foreach (UiSnapshot s in _displayUiSnapshot)
            {
                s.Target.SetActive(s.Active);
                Selectable sel = s.Target.GetComponent<Selectable>();
                if (sel != null) sel.interactable = s.Interactable;
            }
            _displayUiSnapshot.Clear();
            _displayModeActive = false;
            _displayEnterFrame = -1;
            Debug.Log("Display mode OFF: UI restored, rotation stopped.");
        }

        /// <summary>Snapshot the current visibility of every Viewer UI element so exit
        /// can restore it exactly (including conditional ones like Validate being
        /// grayed for non-georeferenced results). Enumerates direct children of
        /// the Viewer canvas so hand-placed buttons/DEM overlays hide too; the
        /// Terrain mesh (has a MeshRenderer) is never hidden.</summary>
        private void CaptureDisplayUi()
        {
            _displayUiSnapshot.Clear();
            foreach (Transform child in transform)
            {
                if (child.GetComponent<MeshRenderer>() != null) continue;
                CaptureUi(child.gameObject);
            }
        }

        private void CaptureUi(GameObject go)
        {
            if (go == null) return;
            Selectable sel = go.GetComponent<Selectable>();
            _displayUiSnapshot.Add(new UiSnapshot
            {
                Target = go,
                Active = go.activeSelf,
                Interactable = sel != null && sel.interactable,
            });
        }

        private void RotateDisplayModel()
        {
            if (_meshGenerator == null) return;
            MeshRenderer mr = _meshGenerator.MeshRenderer;
            if (mr == null) return;
            // Pivot at the mesh's world centre so the Terrain spins in place
            // and stays inside the frame the camera already framed.
            _meshGenerator.transform.RotateAround(
                mr.bounds.center, Vector3.up, _displayRotationSpeed * Time.deltaTime);
        }

        private bool ShouldExitDisplay()
        {
            if (Input.GetMouseButtonDown(0)) return true;
            if (Input.touchCount > 0 && Input.GetTouch(0).phase == TouchPhase.Began) return true;
            return false;
        }

        private void OnToggleWireframe()
        {
            if (_meshGenerator == null) return;
            _meshGenerator.ToggleWireframe();
        }

        private void OnScreenshot()
        {
            string path = $"terrain_{System.DateTime.Now:yyyyMMdd_HHmmss}.png";
            ScreenCapture.CaptureScreenshot(path);
        }

        /// <summary>Job id of the currently loaded result (used for default export filenames).</summary>
        public string JobId => _jobId;

        /// <summary>
        /// Download the job's real DSM as a GeoTIFF (.tif, float32 meters,
        /// correct CRS/transform, no exaggeration; the same true-scale array
        /// the mesh and DEM overlay use) and write it to the caller's chosen
        /// destination path. Requires a georeferenced input - the server
        /// returns a clear error for relative jobs. The GeoTIFF content and
        /// metadata are unchanged; only the destination path became explicit.
        /// Called by DsmExportUI after the native Save As dialog, which then
        /// reports the path via the validation status text.
        /// </summary>
        /// <param name="savePath">Full destination path the user picked.</param>
        public void ExportDsm(string savePath)
        {
            if (string.IsNullOrEmpty(_jobId))
            {
                _validationResultText.text = "DSM export unavailable: no job loaded yet.";
                return;
            }
            if (_launcher == null)
                _launcher = Object.FindFirstObjectByType<Launcher>(FindObjectsInactive.Include);
            if (_launcher == null || _launcher.Api == null)
            {
                _validationResultText.text = "DSM export unavailable: launcher not found.";
                return;
            }
            if (string.IsNullOrEmpty(savePath))
            {
                _validationResultText.text = "DSM export cancelled.";
                return;
            }

            _validationResultText.text = "Exporting DSM...";
            _validationResultText.color = Color.white;

            StartCoroutine(_launcher.Api.ExportDsm(_jobId,
                onSuccess: bytes => SaveDsmFile(bytes, savePath),
                onError: err =>
                {
                    _validationResultText.color = Color.white;
                    _validationResultText.text = $"DSM export error: {err}";
                }));
        }

        private void SaveDsmFile(byte[] bytes, string path)
        {
            File.WriteAllBytes(path, bytes);
            if (_validationResultText != null)
                _validationResultText.text = $"DSM exported:\n{path}";
            Debug.Log($"DSM export saved: {path}");
        }

        private void OnMenu()
        {
            _launcher.ShowMainMenu();
        }
    }
}