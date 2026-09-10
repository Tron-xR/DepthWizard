using UnityEngine;
using UnityEngine.UI;
using TMPro;
using DepthWizard.Core;

namespace DepthWizard.UI
{
    public class MainMenuScreen : MonoBehaviour
    {
        [SerializeField] private Button _newTerrainButton;
        [SerializeField] private Button _settingsButton;
        [SerializeField] private Button _quitButton;
        [SerializeField] private TextMeshProUGUI _serverStatusText;

        private Launcher _launcher;

        private void Awake()
        {
            _launcher = FindObjectOfType<Launcher>();
        }

        private void OnEnable()
        {
            _newTerrainButton.onClick.AddListener(OnNewTerrain);
            _settingsButton.onClick.AddListener(OnSettings);
            _quitButton.onClick.AddListener(OnQuit);
            UpdateServerStatus();
        }

        private void OnDisable()
        {
            _newTerrainButton.onClick.RemoveListener(OnNewTerrain);
            _settingsButton.onClick.RemoveListener(OnSettings);
            _quitButton.onClick.RemoveListener(OnQuit);
        }

        private void UpdateServerStatus()
        {
            if (_launcher == null || _launcher.ServerManager == null) return;

            if (_launcher.ServerManager.IsRunning)
                _serverStatusText.text = "Server: running";
            else
                _serverStatusText.text = "Server: starting...";
        }

        private void OnNewTerrain()
        {
            _launcher.ShowUpload();
        }

        private void OnSettings()
        {
            _launcher.ShowSettings();
        }

        private void OnQuit()
        {
            Application.Quit();
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#endif
        }
    }
}