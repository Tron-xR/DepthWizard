using System;
using UnityEngine;

namespace DepthWizard.Core
{
    public enum ScreenId
    {
        MainMenu,
        Upload,
        Processing,
        Viewer,
        Settings
    }

    public class Launcher : MonoBehaviour
    {
        [Header("Screens (assign in inspector)")]
        [SerializeField] private GameObject _mainMenuScreen;
        [SerializeField] private GameObject _uploadScreen;
        [SerializeField] private GameObject _processingScreen;
        [SerializeField] private GameObject _viewerScreen;
        [SerializeField] private GameObject _settingsScreen;

        [Header("References")]
        [SerializeField] private ServerManager _serverManager;
        [SerializeField] private DepthWizardApi _api;

        private GameObject[] _screens;
        private ScreenId _currentScreen;

        public ServerManager ServerManager => _serverManager;
        public DepthWizardApi Api => _api;

        private void Awake()
        {
            if (_api == null)
                _api = new DepthWizardApi();

            _screens = new GameObject[]
            {
                _mainMenuScreen, _uploadScreen, _processingScreen,
                _viewerScreen, _settingsScreen
            };
        }

        private void Start()
        {
            ShowScreen(ScreenId.MainMenu);

            if (_serverManager != null && !_serverManager.IsRunning)
            {
                StartCoroutine(_serverManager.StartServer(
                    onReady: () => Debug.Log("Server ready"),
                    onError: err => Debug.LogError($"Server error: {err}")
                ));
            }
        }

        public void ShowScreen(ScreenId id)
        {
            for (int i = 0; i < _screens.Length; i++)
            {
                if (_screens[i] != null)
                    _screens[i].SetActive(i == (int)id);
            }
            _currentScreen = id;
        }

        public void ShowMainMenu() => ShowScreen(ScreenId.MainMenu);
        public void ShowUpload() => ShowScreen(ScreenId.Upload);
        public void ShowProcessing() => ShowScreen(ScreenId.Processing);
        public void ShowViewer() => ShowScreen(ScreenId.Viewer);
        public void ShowSettings() => ShowScreen(ScreenId.Settings);
    }
}
