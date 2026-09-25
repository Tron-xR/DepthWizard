using System.Collections;
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

        [Header("Terrain")]
        [SerializeField] private MeshGenerator _meshGenerator;
        [SerializeField] private float _raycastMaxDist = 10000f;

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
            _cam.backgroundColor = new Color(0.85f, 0.85f, 0.85f);
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
            UpdateHUD();
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
            }
            else
            {
                _heightText.text = "Height: --";
                _slopeText.text = "Slope: --";
            }
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

        private void OnMenu()
        {
            _launcher.ShowMainMenu();
        }
    }
}