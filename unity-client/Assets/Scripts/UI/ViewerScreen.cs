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

        private Launcher _launcher;
        private ResultResponse _resultData;
        private string _jobId;
        private UnityEngine.Camera _cam;

        public void SetResultData(ResultResponse data) => _resultData = data;
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

            if (_resultData != null && !_resultData.is_georeferenced)
            {
                _validateButton.interactable = false;
                _validationResultText.text = "Validation needs a georeferenced image (GPS/EXIF tags).";
            }
        }

        private void OnDisable()
        {
            _validateButton.onClick.RemoveListener(OnValidate);
            _wireframeButton.onClick.RemoveListener(OnToggleWireframe);
            _screenshotButton.onClick.RemoveListener(OnScreenshot);
            _menuButton.onClick.RemoveListener(OnMenu);
        }

        private void Start()
        {
            if (_resultData != null)
                StartCoroutine(BuildTerrain());
        }

        private IEnumerator BuildTerrain()
        {
            // Download the heightmap texture
            yield return StartCoroutine(_launcher.Api.DownloadTexture(_resultData.heightmap_url,
                onSuccess: heightTex =>
                {
                    // Download the source texture
                    StartCoroutine(_launcher.Api.DownloadTexture(_resultData.texture_url,
                        onSuccess: sourceTex =>
                        {
                            _meshGenerator.Build(
                                heightTex, sourceTex,
                                _resultData.min_elev, _resultData.max_elev,
                                _resultData.world_width, _resultData.world_depth);
                            FrameTerrain();
                        },
                        onError: err =>
                        {
                            Debug.LogWarning($"Texture download error: {err}");
                            // Build without texture
                            _meshGenerator.Build(
                                heightTex, null,
                                _resultData.min_elev, _resultData.max_elev,
                                _resultData.world_width, _resultData.world_depth);
                            FrameTerrain();
                        }));
                },
                onError: err =>
                {
                    Debug.LogError($"Heightmap download error: {err}");
                }));
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
                _heightText.text = $"Height: {hit.point.y:F1} m";
                _slopeText.text = $"Slope: {Vector3.Angle(hit.normal, Vector3.up):F1}\u00B0";
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

            StartCoroutine(_launcher.Api.Validate(_jobId,
                onSuccess: resp =>
                {
                    _validationResultText.text =
                        $"RMSE: {resp.rmse:F2} m\n" +
                        $"MAE: {resp.mae:F2} m\n" +
                        $"Correlation: {resp.correlation:F3}\n" +
                        $"Reference: {resp.reference_source}";
                },
                onError: err =>
                {
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