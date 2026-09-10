using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace DepthWizard.Core
{
    [Serializable]
    public class UploadResponse
    {
        public string upload_id;
        public bool is_georeferenced;
        public string crs;
        public float[] bounds;
    }

    [Serializable]
    public class ProcessResponse
    {
        public string job_id;
        public string status;
    }

    [Serializable]
    public class StatusResponse
    {
        public string job_id;
        public string status;
        public float progress;
        public string message;
    }

    [Serializable]
    public class ResultResponse
    {
        public string heightmap_url;
        public string texture_url;
        public string dsm_geotiff_url;
        public float min_elev;
        public float max_elev;
        public float cell_size;
        public float world_width;
        public float world_depth;
        public bool is_georeferenced;
    }

    [Serializable]
    public class ValidationResponse
    {
        public string reference_source;
        public string landscape_type;
        public float rmse;
        public float mae;
        public float correlation;
        public string diff_heatmap_url;
    }

    [Serializable]
    public class HealthResponse
    {
        public string status;
    }

    [Serializable]
    public class ErrorResponse
    {
        public string error;
        public string message;
    }

    public class DepthWizardApi
    {
        private readonly string _baseUrl;

        public DepthWizardApi(string host = "127.0.0.1", int port = 8000)
        {
            _baseUrl = $"http://{host}:{port}";
        }

        public IEnumerator Upload(string filePath, Action<UploadResponse> onSuccess, Action<string> onError)
        {
            byte[] fileData = File.ReadAllBytes(filePath);
            string fileName = Path.GetFileName(filePath);
            string mime = GetMimeType(filePath);

            ListFormFile formFile = new ListFormFile(fileName, fileData, mime);
            var sections = new System.Collections.Generic.List<IMultipartFormSection> { formFile };

            using (UnityWebRequest req = UnityWebRequest.Post($"{_baseUrl}/upload", sections))
            {
                req.timeout = 120;
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    onError?.Invoke(req.error);
                    yield break;
                }

                var resp = JsonUtility.FromJson<UploadResponse>(req.downloadHandler.text);
                onSuccess?.Invoke(resp);
            }
        }

        public IEnumerator Process(string uploadId, Action<ProcessResponse> onSuccess, Action<string> onError)
        {
            using (UnityWebRequest req = new UnityWebRequest($"{_baseUrl}/process/{uploadId}", "POST"))
            {
                req.uploadHandler = new UploadHandlerRaw(new byte[0]);
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");
                req.timeout = 30;
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    onError?.Invoke(req.error);
                    yield break;
                }

                var resp = JsonUtility.FromJson<ProcessResponse>(req.downloadHandler.text);
                onSuccess?.Invoke(resp);
            }
        }

        public IEnumerator GetStatus(string jobId, Action<StatusResponse> onSuccess, Action<string> onError)
        {
            using (UnityWebRequest req = UnityWebRequest.Get($"{_baseUrl}/status/{jobId}"))
            {
                req.timeout = 10;
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    onError?.Invoke(req.error);
                    yield break;
                }

                var resp = JsonUtility.FromJson<StatusResponse>(req.downloadHandler.text);
                onSuccess?.Invoke(resp);
            }
        }

        public IEnumerator GetResult(string jobId, Action<ResultResponse> onSuccess, Action<string> onError)
        {
            using (UnityWebRequest req = UnityWebRequest.Get($"{_baseUrl}/result/{jobId}"))
            {
                req.timeout = 10;
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    onError?.Invoke(req.error);
                    yield break;
                }

                var resp = JsonUtility.FromJson<ResultResponse>(req.downloadHandler.text);
                onSuccess?.Invoke(resp);
            }
        }

        public IEnumerator Validate(string jobId, Action<ValidationResponse> onSuccess, Action<string> onError)
        {
            using (UnityWebRequest req = UnityWebRequest.Get($"{_baseUrl}/validate/{jobId}"))
            {
                req.timeout = 120;
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    onError?.Invoke(ParseErrorMessage(req.downloadHandler?.text) ?? req.error);
                    yield break;
                }

                var resp = JsonUtility.FromJson<ValidationResponse>(req.downloadHandler.text);
                onSuccess?.Invoke(resp);
            }
        }

        private static string ParseErrorMessage(string body)
        {
            if (string.IsNullOrEmpty(body)) return null;
            try
            {
                var detail = JsonUtility.FromJson<ErrorDetail>(body);
                return detail?.detail?.message;
            }
            catch
            {
                return null;
            }
        }

        [Serializable]
        private class ErrorDetail
        {
            public Detail detail;
        }

        [Serializable]
        private class Detail
        {
            public string error;
            public string message;
        }

        public IEnumerator HealthCheck(Action<HealthResponse> onSuccess, Action<string> onError)
        {
            using (UnityWebRequest req = UnityWebRequest.Get($"{_baseUrl}/health"))
            {
                req.timeout = 5;
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    onError?.Invoke(req.error);
                    yield break;
                }

                var resp = JsonUtility.FromJson<HealthResponse>(req.downloadHandler.text);
                onSuccess?.Invoke(resp);
            }
        }

        public IEnumerator DownloadTexture(string relativeUrl, Action<Texture2D> onSuccess, Action<string> onError)
        {
            string url = _baseUrl + relativeUrl;
            using (UnityWebRequest req = UnityWebRequestTexture.GetTexture(url))
            {
                req.timeout = 60;
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    onError?.Invoke(req.error);
                    yield break;
                }

                Texture2D tex = DownloadHandlerTexture.GetContent(req);
                onSuccess?.Invoke(tex);
            }
        }

        private string GetMimeType(string path)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            switch (ext)
            {
                case ".png":  return "image/png";
                case ".jpg":  return "image/jpeg";
                case ".jpeg": return "image/jpeg";
                case ".tif":  return "image/tiff";
                case ".tiff": return "image/tiff";
                default:      return "application/octet-stream";
            }
        }

        private class ListFormFile : IMultipartFormSection
        {
            private string _fileName;
            private byte[] _data;
            private string _contentType;

            public string sectionName => "file";
            public string fileName => _fileName;
            public string contentType => _contentType;
            public byte[] sectionData => _data;

            public ListFormFile(string fileName, byte[] data, string contentType)
            {
                _fileName = fileName;
                _data = data;
                _contentType = contentType;
            }
        }
    }
}
