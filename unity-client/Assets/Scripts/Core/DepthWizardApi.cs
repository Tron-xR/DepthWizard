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
        // Optional diagnostic scatter (reference vs prediction density).
        public string scatter_url;
        // Active depth backend slug that produced the validated DSM.
        public string backend;
        // Additive calibration degeneracy flag from the server: true for flat /
        // near-degenerate tiles. The job itself still completed normally.
        public bool degenerate_calibration;
        public string calibration_reason;
        // Uncalibrated model polarity fields (server /validate + /evaluate).
        // raw_correlation_signed is the held-out raw correlation BEFORE any
        // calibration sign flip; scale_sign is "+" / "-" of the fitted scale.
        // polarity_inverted true = model signal clearly inverted. All Optional
        // on the server: JsonUtility maps missing/null to defaults, so the
        // has* presence flags below (set after deserialization) decide whether
        // a line is shown at all.
        public float raw_correlation_signed;
        public string scale_sign;
        public bool polarity_inverted;
        public string polarity_reason;
        // Calibration fit parameters (server always reports them for absolute
        // jobs; the client displays them in the results summary).
        public float calibration_scale;
        public float calibration_offset;
        // Additive summary stats on the SAME scored pixels as rmse/mae/
        // correlation (held-out 20%, or whole grid for legacy jobs), in
        // meters: bias = mean(prediction - reference), positive = overpredict.
        public float bias;
        public float prediction_std;
        public float reference_std;
        [System.NonSerialized] public bool hasRawCorrelation;
        [System.NonSerialized] public bool hasPolarityInverted;
        [System.NonSerialized] public bool hasPolarityReason;
        [System.NonSerialized] public bool hasCorrelation;
        [System.NonSerialized] public bool hasCalibrationScale;
        [System.NonSerialized] public bool hasCalibrationOffset;
        [System.NonSerialized] public bool hasBias;
        [System.NonSerialized] public bool hasPredictionStd;
        [System.NonSerialized] public bool hasReferenceStd;
        [System.NonSerialized] public bool hasBackend;
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

                string body = req.downloadHandler.text;
                var resp = JsonUtility.FromJson<ValidationResponse>(body);
                // Older servers / Optional fields that serialize to null become
                // type defaults under JsonUtility (0 / false / ""); record which
                // polarity fields were actually present with a real value so the
                // UI can omit those lines instead of showing "0.000".
                if (body != null)
                {
                    resp.hasRawCorrelation = HasNonNullField(body, "raw_correlation_signed");
                    resp.hasPolarityInverted = HasNonNullField(body, "polarity_inverted");
                    resp.hasPolarityReason = HasNonNullField(body, "polarity_reason");
                    resp.hasCorrelation = HasNonNullField(body, "correlation");
                    resp.hasCalibrationScale = HasNonNullField(body, "calibration_scale");
                    resp.hasCalibrationOffset = HasNonNullField(body, "calibration_offset");
                    resp.hasBias = HasNonNullField(body, "bias");
                    resp.hasPredictionStd = HasNonNullField(body, "prediction_std");
                    resp.hasReferenceStd = HasNonNullField(body, "reference_std");
                    resp.hasBackend = HasNonNullField(body, "backend");
                }
                onSuccess?.Invoke(resp);
            }
        }

        public IEnumerator ExportDsm(string jobId, Action<byte[]> onSuccess, Action<string> onError)
        {
            using (UnityWebRequest req = UnityWebRequest.Get($"{_baseUrl}/export-dsm/{jobId}"))
            {
                req.timeout = 60;
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    onError?.Invoke(ParseErrorMessage(req.downloadHandler?.text) ?? req.error);
                    yield break;
                }

                onSuccess?.Invoke(req.downloadHandler.data);
            }
        }
        public IEnumerator ExportDem(string jobId, Action<byte[]> onSuccess, Action<string> onError)
        {
            using (UnityWebRequest req = UnityWebRequest.Get($"{_baseUrl}/export-dem/{jobId}"))
            {
                req.timeout = 60;
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    onError?.Invoke(ParseErrorMessage(req.downloadHandler?.text) ?? req.error);
                    yield break;
                }

                onSuccess?.Invoke(req.downloadHandler.data);
            }
        }

        // True when body contains '"name": <non-null>' for the given JSON field.
        private static bool HasNonNullField(string body, string name)
        {
            string key = "\"" + name + "\"";
            int idx = body.IndexOf(key, StringComparison.Ordinal);
            while (idx >= 0)
            {
                int colon = body.IndexOf(':', idx + key.Length);
                if (colon >= 0)
                {
                    int i = colon + 1;
                    while (i < body.Length && (body[i] == ' ' || body[i] == '\t')) i++;
                    if (i < body.Length && body[i] != 'n')
                        return true;
                }
                idx = body.IndexOf(key, idx + key.Length, StringComparison.Ordinal);
            }
            return false;
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
