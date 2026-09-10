using UnityEngine;
using UnityEngine.UI;
using TMPro;
using DepthWizard.Core;

namespace DepthWizard.UI
{
    public class SettingsScreen : MonoBehaviour
    {
        [SerializeField] private Button _backButton;
        [SerializeField] private Button _checkHealthButton;
        [SerializeField] private TextMeshProUGUI _healthStatusText;
        [SerializeField] private TextMeshProUGUI _modelInfoText;

        private Launcher _launcher;

        private void Awake()
        {
            _launcher = FindObjectOfType<Launcher>();
        }

        private void OnEnable()
        {
            _backButton.onClick.AddListener(OnBack);
            _checkHealthButton.onClick.AddListener(OnCheckHealth);
            _modelInfoText.text = "Model: IMELE (building heights)\n" +
                                  "Server: FastAPI + uvicorn\n" +
                                  "GPU: Check Task Manager";
            _healthStatusText.text = "";
        }

        private void OnDisable()
        {
            _backButton.onClick.RemoveListener(OnBack);
            _checkHealthButton.onClick.RemoveListener(OnCheckHealth);
        }

        private void OnCheckHealth()
        {
            _healthStatusText.text = "Checking...";

            StartCoroutine(_launcher.ServerManager.CheckHealth(
                alive =>
                {
                    _healthStatusText.text = alive
                        ? "<color=green>Server is running</color>"
                        : "<color=red>Server not responding</color>";
                },
                err =>
                {
                    _healthStatusText.text = $"<color=red>Error: {err}</color>";
                }));
        }

        private void OnBack()
        {
            _launcher.ShowMainMenu();
        }
    }
}