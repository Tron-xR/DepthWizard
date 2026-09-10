using System;
using System.Diagnostics;
using System.IO;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

namespace DepthWizard.Core
{
    public class ServerManager : MonoBehaviour
    {
        [SerializeField] private int _port = 8000;
        [SerializeField] private float _healthCheckInterval = 0.5f;
        [SerializeField] private int _maxHealthRetries = 60;

        private Process _serverProcess;
        private bool _isRunning;
        private string _serverUrl;

        public bool IsRunning => _isRunning;
        public string ServerUrl => _serverUrl;

        private void Awake()
        {
            _serverUrl = $"http://127.0.0.1:{_port}";
            DontDestroyOnLoad(gameObject);
        }

        public IEnumerator StartServer(Action onReady, Action<string> onError)
        {
            string serverPath = FindServerExecutable();

            if (string.IsNullOrEmpty(serverPath))
            {
                onError?.Invoke("Could not locate the Python server executable. " +
                    "Ensure 'server' is in StreamingAssets or a sibling directory.");
                yield break;
            }

            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = serverPath,
                    Arguments = $"-m uvicorn app.main:app --host 127.0.0.1 --port {_port}",
                    WorkingDirectory = Path.GetDirectoryName(serverPath),
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true
                };

                _serverProcess = Process.Start(psi);
                _isRunning = true;
            }
            catch (Exception e)
            {
                onError?.Invoke($"Failed to start server: {e.Message}");
                yield break;
            }

            for (int i = 0; i < _maxHealthRetries; i++)
            {
                yield return new WaitForSeconds(_healthCheckInterval);

                bool alive = false;
                yield return CheckHealth(alive_ => alive = alive_, _ => { });

                if (alive)
                {
                    onReady?.Invoke();
                    yield break;
                }
            }

            onError?.Invoke("Server failed to respond after maximum retries.");
            StopServer();
        }

        public IEnumerator CheckHealth(Action<bool> callback, Action<string> onError)
        {
            using (UnityWebRequest req = UnityWebRequest.Get($"{_serverUrl}/health"))
            {
                req.timeout = 3;
                req.downloadHandler = new DownloadHandlerBuffer();
                yield return req.SendWebRequest();

                if (req.result == UnityWebRequest.Result.Success)
                {
                    var resp = JsonUtility.FromJson<HealthResponse>(req.downloadHandler.text);
                    callback?.Invoke(resp.status == "ok");
                }
                else
                {
                    callback?.Invoke(false);
                }
            }
        }

        public void StopServer()
        {
            if (_serverProcess != null && !_serverProcess.HasExited)
            {
                try { _serverProcess.Kill(); } catch { }
                _serverProcess.Dispose();
                _serverProcess = null;
            }
            _isRunning = false;
        }

        private string FindServerExecutable()
        {
            // Candidate locations relative to the application
            string[] candidates = new string[]
            {
                // Packaged frozen server: <data>/StreamingAssets/server/depthwizard-server/.
                // Application.dataPath is <project>/Assets in the Editor and
                // <player>/DepthWizard_Data in a build; StreamingAssets sits under
                // both, so this one candidate resolves correctly in either case.
                Path.Combine(Application.dataPath, "StreamingAssets", "server",
                    "depthwizard-server", "depthwizard-server.exe"),
                Path.Combine(Application.dataPath, "..", "StreamingAssets", "server", "app", "main.py"),
                Path.Combine(Application.dataPath, "..", "..", "server", ".venv", "Scripts", "python.exe"),
                Path.Combine(Application.dataPath, "..", "..", "server", ".venv", "bin", "python"),
            };

            foreach (string path in candidates)
            {
                if (File.Exists(path)) return path;
            }

            // Check if `python` is on PATH and the server app.main module is importable
            string fallbackPython = SystemInfo.operatingSystemFamily == OperatingSystemFamily.Windows
                ? "python" : "python3";
            return fallbackPython;
        }

        private void OnApplicationQuit()
        {
            StopServer();
        }

        private class HealthResponse
        {
            public string status;
        }
    }
}
