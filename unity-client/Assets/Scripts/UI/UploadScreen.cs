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

        private Launcher _launcher;
        private string _selectedPath;
        private UploadResponse _lastUpload;

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
        }

        private void OnDisable()
        {
            _browseButton.onClick.RemoveListener(OnBrowse);
            _processButton.onClick.RemoveListener(OnProcess);
            _backButton.onClick.RemoveListener(OnBack);
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