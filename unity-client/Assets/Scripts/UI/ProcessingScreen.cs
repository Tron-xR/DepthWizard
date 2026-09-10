using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using DepthWizard.Core;

namespace DepthWizard.UI
{
    public class ProcessingScreen : MonoBehaviour
    {
        [SerializeField] private Slider _progressBar;
        [SerializeField] private TextMeshProUGUI _stageText;
        [SerializeField] private TextMeshProUGUI _progressText;
        [SerializeField] private Button _cancelButton;

        private Launcher _launcher;
        private string _jobId;
        private bool _polling;
        private Coroutine _pollCoroutine;

        /// <summary>
        /// Set externally by UploadScreen before calling ShowProcessing().
        /// </summary>
        public string PendingJobId { get; set; }

        private void Awake()
        {
            _launcher = FindObjectOfType<Launcher>();
        }

        private void OnEnable()
        {
            _cancelButton.onClick.AddListener(OnCancel);
            _stageText.text = "Uploading...";
            _progressText.text = "0%";
            _progressBar.value = 0f;

            if (!string.IsNullOrEmpty(PendingJobId))
            {
                _jobId = PendingJobId;
                PendingJobId = null;
                _polling = true;
                _pollCoroutine = StartCoroutine(PollStatus());
            }
        }

        private void OnDisable()
        {
            _cancelButton.onClick.RemoveListener(OnCancel);
            _polling = false;
            if (_pollCoroutine != null) StopCoroutine(_pollCoroutine);
        }

        public void StartPolling(string jobId)
        {
            _jobId = jobId;
            _polling = true;
            _pollCoroutine = StartCoroutine(PollStatus());
        }

        private IEnumerator PollStatus()
        {
            while (_polling)
            {
                yield return new WaitForSeconds(0.5f);

                yield return StartCoroutine(_launcher.Api.GetStatus(_jobId,
onSuccess: resp =>
                {
                    Debug.Log($"Poll {_jobId} -> {resp.status} @ {resp.progress * 100f:F0}%");
                    _progressBar.value = resp.progress;
                        _progressText.text = $"{resp.progress * 100f:F0}%";

                        switch (resp.status)
                        {
                            case "queued":
                                _stageText.text = "Queued...";
                                break;
                            case "running":
                                _stageText.text = FormatStage(resp.progress);
                                break;
                            case "done":
                                _polling = false;
                                OnJobComplete();
                                break;
                            case "failed":
                                _polling = false;
                                _stageText.text = $"Failed: {resp.message}";
                                break;
                        }
                    },
                    onError: err =>
                    {
                        Debug.LogError($"Status poll error for job {_jobId}: {err}");
                    }));
            }
        }

        private string FormatStage(float progress)
        {
            if (progress < 0.2f) return "Running depth model...";
            if (progress < 0.6f) return "Generating heightmap...";
            if (progress < 0.8f) return "Building terrain mesh...";
            return "Finalizing...";
        }

        private void OnJobComplete()
        {
            _stageText.text = "Complete! Loading terrain...";
            StartCoroutine(LoadResult());
        }

        private IEnumerator LoadResult()
        {
            yield return StartCoroutine(_launcher.Api.GetResult(_jobId,
                onSuccess: resp =>
                {
                    var viewer = UnityEngine.Object.FindFirstObjectByType<ViewerScreen>(FindObjectsInactive.Include);
                    if (viewer != null)
                    {
                        viewer.SetResultData(resp);
                        viewer.SetJobId(_jobId);
                    }
                    _launcher.ShowViewer();
                },
                onError: err =>
                {
                    _stageText.text = $"Error loading result: {err}";
                }));
        }

        private void OnCancel()
        {
            _polling = false;
            _launcher.ShowUpload();
        }
    }
}