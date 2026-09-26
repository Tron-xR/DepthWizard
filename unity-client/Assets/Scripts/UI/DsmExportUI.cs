using System;
using UnityEngine;
using UnityEngine.UI;
using SFB;

namespace DepthWizard.UI
{
    /// <summary>
    /// Save-dialog flow for the Export DSM button:
    /// - hidden on Start, revealed only after the DEM View button is clicked
    ///   (DEM View's existing onClick listeners are kept; this only adds to them);
    /// - on click, opens a native OS Save As dialog pre-filled with the old
    ///   dsm_&lt;jobid&gt;_&lt;timestamp&gt;.tif name and writes the GeoTIFF to the
    ///   user-chosen path by calling ViewerScreen.ExportDsm(savePath).
    /// Replaces the previous fixed AppData-path write. Wired in the Inspector.
    /// </summary>
    public class DsmExportUI : MonoBehaviour
    {
        [SerializeField] private Button _exportDsmButton;
        [SerializeField] private Button _demViewButton;
        [SerializeField] private ViewerScreen _viewerScreen;

        private void Start()
        {
            if (_exportDsmButton == null)
            {
                Debug.LogError("DsmExportUI: _exportDsmButton is not assigned.");
                return;
            }
            if (_demViewButton == null)
            {
                Debug.LogError("DsmExportUI: _demViewButton is not assigned.");
                return;
            }
            if (_viewerScreen == null)
            {
                Debug.LogError("DsmExportUI: _viewerScreen is not assigned.");
                return;
            }

            _exportDsmButton.gameObject.SetActive(false);
            _demViewButton.onClick.AddListener(RevealExportDsm);
            _exportDsmButton.onClick.AddListener(OnExportDsmClicked);
        }

        private void RevealExportDsm()
        {
            _exportDsmButton.gameObject.SetActive(true);
        }

        private void OnExportDsmClicked()
        {
            string defaultName = string.IsNullOrEmpty(_viewerScreen.JobId)
                ? $"dsm_{DateTime.Now:yyyyMMdd_HHmmss}.tif"
                : $"dsm_{_viewerScreen.JobId}_{DateTime.Now:yyyyMMdd_HHmmss}.tif";

            string savePath = StandaloneFileBrowser.SaveFilePanel(
                "Save DSM", "", defaultName,
                new[] { new ExtensionFilter("GeoTIFF", "tif", "tiff") });

            if (string.IsNullOrEmpty(savePath))
                return;

            _viewerScreen.ExportDsm(savePath);
        }
    }
}