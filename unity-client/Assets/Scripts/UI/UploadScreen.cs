using System;
using System.Collections;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using DepthWizard.Core;

namespace DepthWizard.UI
{
    public class UploadScreen : MonoBehaviour
    {
        [SerializeField] private Button _browseButton;
        [SerializeField] private Button _processButton;
        [SerializeField] private Button _backButton;
        [SerializeField] private TextMeshProUGUI _filePathText;
        [SerializeField] private TextMeshProUGUI _geoInfoText;
        [SerializeField] private TextMeshProUGUI _errorText;

        private const int MaxPreviewDim = 512;
        private const float PreviewBoxSize = 240f;
        private const float PreviewX = -390f;
        private const float PreviewY = 109f;

        private Launcher _launcher;
        private string _selectedPath;
        private UploadResponse _lastUpload;
        private RawImage _previewImage;
        private Text _previewNote;
        private Texture2D _previewTex;

        private void Awake()
        {
            _launcher = FindObjectOfType<Launcher>();
        }

        private void OnEnable()
        {
            _browseButton.onClick.AddListener(OnBrowse);
            _processButton.onClick.AddListener(OnProcess);
            _backButton.onClick.AddListener(OnBack);
            _processButton.interactable = false;
            _errorText.text = "";
            _geoInfoText.text = "";
            _filePathText.text = "No file selected";

            EnsurePreview();
            ClearPreview();
        }

        private void OnDisable()
        {
            _browseButton.onClick.RemoveListener(OnBrowse);
            _processButton.onClick.RemoveListener(OnProcess);
            _backButton.onClick.RemoveListener(OnBack);
            ClearPreview();
        }

        private void OnBrowse()
        {
            _browseButton.interactable = false;
            StartCoroutine(PickFileAndUpload());
        }

        private IEnumerator PickFileAndUpload()
        {
            Debug.Log("[picker] coroutine started");
            yield return new WaitForEndOfFrame();

            string path;
            try
            {
                path = OpenFilePicker();
            }
            catch (Exception e)
            {
                Debug.LogWarning($"File picker failed: {e}");
                path = null;
            }
            _browseButton.interactable = true;

            if (string.IsNullOrEmpty(path)) yield break;

            _selectedPath = path;
            _filePathText.text = Path.GetFileName(path);
            _errorText.text = "";
            _processButton.interactable = false;

            UploadFile(path);
            ShowPreview(path);
        }

        private void UploadFile(string path)
        {
            StartCoroutine(_launcher.Api.Upload(path,
                onSuccess: resp =>
                {
                    _lastUpload = resp;
                    string geoLabel = resp.is_georeferenced
                        ? $"Georeferenced \u2014 CRS: {resp.crs}"
                        : "No geo metadata \u2014 relative mode";
                    _geoInfoText.text = geoLabel;
                    _processButton.interactable = true;

                    if (IsTiff(path))
                        StartCoroutine(ShowServerPreview(resp.upload_id));
                },
                onError: err =>
                {
                    _errorText.text = $"Upload failed: {err}";
                    _processButton.interactable = false;
                }));
        }

        private void OnProcess()
        {
            if (_lastUpload == null) return;

            _processButton.interactable = false;
            _errorText.text = "";

            StartCoroutine(_launcher.Api.Process(_lastUpload.upload_id,
                onSuccess: resp =>
                {
                    Debug.Log($"Process returned job_id={resp.job_id}");
                    var processing = UnityEngine.Object.FindFirstObjectByType<ProcessingScreen>(FindObjectsInactive.Include);
                    if (processing != null)
                        processing.PendingJobId = resp.job_id;
                    _launcher.ShowProcessing();
                },
                onError: err =>
                {
                    _errorText.text = $"Process error: {err}";
                    _processButton.interactable = true;
                }));
        }

        private void OnBack()
        {
            _launcher.ShowMainMenu();
        }

        #region Thumbnail preview

        private void EnsurePreview()
        {
            if (_previewImage != null) return;

            var go = new GameObject("ThumbnailPreview",
                typeof(RectTransform), typeof(CanvasRenderer));
            RectTransform rt = go.GetComponent<RectTransform>();
            rt.SetParent(transform, false);
            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.pivot = new Vector2(0.5f, 0.5f);
            rt.anchoredPosition = new Vector2(PreviewX, PreviewY);
            rt.sizeDelta = new Vector2(PreviewBoxSize, PreviewBoxSize);

            _previewImage = go.AddComponent<RawImage>();
            _previewImage.color = new Color(0.85f, 0.85f, 0.85f, 1f);
            _previewImage.raycastTarget = false;
            go.SetActive(false);

            var caption = new GameObject("PreviewNote",
                typeof(RectTransform), typeof(CanvasRenderer));
            RectTransform crt = caption.GetComponent<RectTransform>();
            crt.SetParent(go.transform, false);
            crt.anchorMin = new Vector2(0f, 0f);
            crt.anchorMax = new Vector2(1f, 0f);
            crt.pivot = new Vector2(0.5f, 0f);
            crt.anchoredPosition = new Vector2(0f, -18f);
            crt.sizeDelta = new Vector2(0f, 18f);

            _previewNote = caption.AddComponent<Text>();
            _previewNote.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            _previewNote.fontSize = 13;
            _previewNote.color = new Color(0.55f, 0.35f, 0.35f, 1f);
            _previewNote.alignment = TextAnchor.MiddleCenter;
            _previewNote.raycastTarget = false;
            caption.SetActive(false);
        }

        private void ClearPreview()
        {
            if (_previewTex != null)
            {
                Destroy(_previewTex);
                _previewTex = null;
            }
            if (_previewImage != null)
            {
                _previewImage.texture = null;
                _previewImage.rectTransform.sizeDelta = new Vector2(PreviewBoxSize, PreviewBoxSize);
                _previewImage.gameObject.SetActive(false);
            }
            if (_previewNote != null)
                _previewNote.gameObject.SetActive(false);
        }

        private void ShowPreview(string path)
        {
            if (_previewImage == null || IsTiff(path)) return;
            SetPreviewTexture(LoadThumbnail(path));
        }

        private void SetPreviewTexture(Texture2D tex)
        {
            if (tex == null)
            {
                ClearPreview();
                return;
            }
            if (_previewTex != null)
            {
                Destroy(_previewTex);
                _previewTex = null;
            }
            _previewTex = tex;
            if (_previewImage == null) return;
            _previewNote?.gameObject.SetActive(false);
            _previewImage.texture = tex;
            _previewImage.rectTransform.sizeDelta = FitInto(tex.width, tex.height, PreviewBoxSize);
            _previewImage.gameObject.SetActive(true);
        }

        private void ShowPreviewUnavailable()
        {
            if (_previewTex != null)
            {
                Destroy(_previewTex);
                _previewTex = null;
            }
            if (_previewImage == null) return;
            _previewImage.texture = null;
            _previewImage.rectTransform.sizeDelta = new Vector2(PreviewBoxSize, PreviewBoxSize);
            _previewImage.gameObject.SetActive(true);
            if (_previewNote != null)
            {
                _previewNote.text = "Preview unavailable";
                _previewNote.gameObject.SetActive(true);
            }
        }

        /// <summary>
        /// GeoTIFF can't be decoded client-side (Unity's ImageConversion is
        /// PNG/JPG only), so its preview is rendered server-side once the
        /// upload is ingested: GET /preview/{upload_id} returns a low-res PNG.
        /// Purely visual; the upload/Process flow never depends on this.
        /// </summary>
        private IEnumerator ShowServerPreview(string uploadId)
        {
            yield return _launcher.Api.DownloadTexture("/preview/" + uploadId,
                onSuccess: tex =>
                {
                    SetPreviewTexture(tex);
                },
                onError: err =>
                {
                    Debug.LogWarning($"[preview] server render failed for {uploadId}: {err}");
                    ShowPreviewUnavailable();
                });
        }

        private static bool IsTiff(string path)
        {
            string ext = Path.GetExtension(path);
            return ext != null && (ext.Equals(".tif", StringComparison.OrdinalIgnoreCase)
                                   || ext.Equals(".tiff", StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Load PNG/JPG bytes into a Texture2D, downsampling anything larger
        /// than MaxPreviewDim so a huge source never allocates a full-res
        /// texture just for the thumbnail. Returns null for unsupported or
        /// corrupt files (caller hides the preview; picker flow unaffected).
        /// </summary>
        private static Texture2D LoadThumbnail(string path)
        {
            byte[] bytes;
            try
            {
                bytes = File.ReadAllBytes(path);
            }
            catch (IOException e)
            {
                Debug.LogWarning($"[preview] cannot read {path}: {e.Message}");
                return null;
            }

            var full = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            if (!ImageConversion.LoadImage(full, bytes))
            {
                Debug.LogWarning($"[preview] unreadable image format: {path}");
                Destroy(full);
                return null;
            }

            int maxDim = Mathf.Max(full.width, full.height);
            if (maxDim <= MaxPreviewDim) return full;

            int w = Mathf.RoundToInt((float)full.width * MaxPreviewDim / maxDim);
            int h = Mathf.RoundToInt((float)full.height * MaxPreviewDim / maxDim);

            var rt = RenderTexture.GetTemporary(w, h, 0);
            var thumb = new Texture2D(w, h, TextureFormat.RGBA32, false);
            try
            {
                RenderTexture.active = rt;
                Graphics.Blit(full, rt);
                thumb.ReadPixels(new Rect(0, 0, w, h), 0, 0);
                thumb.Apply();
            }
            finally
            {
                RenderTexture.active = null;
                RenderTexture.ReleaseTemporary(rt);
                Destroy(full);
            }
            return thumb;
        }

        private static Vector2 FitInto(int w, int h, float box)
        {
            if (w <= 0 || h <= 0) return new Vector2(box, box);
            float s = box / Mathf.Max(w, h);
            return new Vector2(w * s, h * s);
        }

        #endregion

        #region Native file picker

        private static string OpenFilePicker()
        {
#if UNITY_EDITOR
            return UnityEditor.EditorUtility.OpenFilePanel(
                "Select an image or GeoTIFF", "", "png;jpg;jpeg;tif;tiff");
#elif UNITY_STANDALONE_WIN
            string path = null;
            var t = new Thread(() => path = OpenNativePicker());
            t.SetApartmentState(ApartmentState.STA);
            t.Start();
            t.Join();
            return path;
#else
            return null;
#endif
        }

#if UNITY_STANDALONE_WIN
        private const uint FOS_FORCEFILESYSTEM = 0x00000040;
        private const uint SIGDN_FILESYSPATH = 0x80058000;
        private static readonly Guid CLSID_FileOpenDialog =
            new Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7");
        private static readonly Guid IID_IFileDialog =
            new Guid("42F85136-DB7E-439C-85F1-E4075D135FC8");
        private static readonly Guid IID_IShellItem =
            new Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe");

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct COMDLG_FILTERSPEC
        {
            public string pszName;
            public string pszSpec;
        }

        [ComImport, Guid("42F85136-DB7E-439C-85F1-E4075D135FC8"),
         InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        private interface IFileDialog
        {
            [PreserveSig] int Show(IntPtr owner);
            [PreserveSig] int SetFileTypes(uint cFileTypes,
                [MarshalAs(UnmanagedType.LPArray)] COMDLG_FILTERSPEC[] rgFilterSpec);
            [PreserveSig] int SetFileTypeIndex(uint iFileType);
            [PreserveSig] int GetFileTypeIndex(out uint piFileType);
            [PreserveSig] int Advise(IntPtr pfn, out uint pdwCookie);
            [PreserveSig] int Unadvise(uint dwCookie);
            [PreserveSig] int SetOptions(uint fos);
            [PreserveSig] int GetOptions(out uint pfos);
            [PreserveSig] int SetDefaultFolder(IntPtr psi);
            [PreserveSig] int SetFolder(IntPtr psi);
            [PreserveSig] int GetFolder(out IntPtr ppsi);
            [PreserveSig] int GetCurrentSelection(out IntPtr ppsi);
            [PreserveSig] int SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
            [PreserveSig] int GetFileName(out IntPtr pszName);
            [PreserveSig] int SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
            [PreserveSig] int SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
            [PreserveSig] int SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
            [PreserveSig] int GetResult(out IShellItem ppsi);
            [PreserveSig] int AddPlace(IntPtr psi, int fdap);
            [PreserveSig] int SetDefaultExtension(
                [MarshalAs(UnmanagedType.LPWStr)] string pszDefaultExtension);
            [PreserveSig] int Close(int hr);
            [PreserveSig] int SetClientGuid(ref Guid guid);
            [PreserveSig] int ClearClientData();
            [PreserveSig] int SetFilter(IntPtr pFilter);
        }

        [ComImport, Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe"),
         InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        private interface IShellItem
        {
            [PreserveSig] int BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
            [PreserveSig] int GetParent(out IShellItem ppsi);
            [PreserveSig] int GetDisplayName(uint sigdnName,
                [MarshalAs(UnmanagedType.LPWStr)] out string ppszName);
            [PreserveSig] int GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
            [PreserveSig] int Compare(IShellItem psi, uint hint, out int piOrder);
        }

        [DllImport("ole32.dll")]
        private static extern int CoInitializeEx(IntPtr pvReserved, uint dwCoInit);

        [DllImport("ole32.dll")]
        private static extern void CoUninitialize();

        private static string OpenNativePicker()
        {
            Debug.Log("[picker] opening IFileDialog");
            CoInitializeEx(IntPtr.Zero, 0);
            try
            {
                Type dialogType = Type.GetTypeFromCLSID(CLSID_FileOpenDialog);
                var dlg = (IFileDialog)Activator.CreateInstance(dialogType);
                dlg.SetOptions(FOS_FORCEFILESYSTEM);
                dlg.SetTitle("Select an image or GeoTIFF");
                dlg.SetFileTypes(2, new[]
                {
                    new COMDLG_FILTERSPEC
                    {
                        pszName = "Image files",
                        pszSpec = "*.png;*.jpg;*.jpeg;*.tif;*.tiff"
                    },
                    new COMDLG_FILTERSPEC { pszName = "All files", pszSpec = "*.*" }
                });
                dlg.SetFileTypeIndex(1);
                if (dlg.Show(IntPtr.Zero) < 0) return null;
                IShellItem item;
                if (dlg.GetResult(out item) < 0) return null;
                string selected;
                if (item.GetDisplayName(SIGDN_FILESYSPATH, out selected) < 0) return null;
                return selected;
            }
            finally
            {
                CoUninitialize();
            }
        }
#endif

        #endregion
    }
}