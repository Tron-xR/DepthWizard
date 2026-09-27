using System;
using UnityEngine;
using UnityEngine.UI;
using SFB;

namespace DepthWizard.UI
{
    /// <summary>
    /// Save-dialog flow for the Export DEM button (bare-earth DEM - buildings,
    /// trees and vegetation removed by the server's progressive morphological
    /// filter, a genuinely separate surface from the Export DSM button):
    /// - hidden on Start, revealed together with Export DSM after the DEM View
    ///   button is clicked;
    /// - on click, opens a native OS Save As dialog pre-filled with the
    ///   dem_&lt;jobid&gt;_&lt;timestamp&gt;.tif name and writes the filtered GeoTIFF
    ///   to the user-chosen path by calling ViewerScreen.ExportDem(savePath).
    /// Separate component from DsmExportUI because it exports different data.
    /// </summary>
    public class DemExportUI : MonoBehaviour
    {
        [SerializeField] private Button _exportDemButton;
        [SerializeField] private Button _demViewButton;
        [SerializeField] private ViewerScreen _viewerScreen;

        private void Start()
        {
            if (_exportDemButton == null)
            {
                Debug.LogError("DemExportUI: _exportDemButton is not assigned.");
                return;
            }
            if (_demViewButton == null)
            {
                Debug.LogError("DemExportUI: _demViewButton is not assigned.");
                return;
            }
            if (_viewerScreen == null)
            {
                Debug.LogError("DemExportUI: _viewerScreen is not assigned.");
                return;
            }

            _exportDemButton.gameObject.SetActive(false);
            _demViewButton.onClick.AddListener(RevealExportDem);
            _exportDemButton.onClick.AddListener(OnExportDemClicked);
        }

        private void RevealExportDem()
        {
            _exportDemButton.gameObject.SetActive(true);
        }

        private void OnExportDemClicked()
        {
            string defaultName = string.IsNullOrEmpty(_viewerScreen.JobId)
                ? $"dem_{DateTime.Now:yyyyMMdd_HHmmss}.tif"
                : $"dem_{_viewerScreen.JobId}_{DateTime.Now:yyyyMMdd_HHmmss}.tif";

            string savePath = StandaloneFileBrowser.SaveFilePanel(
                "Save DEM", "", defaultName,
                new[] { new ExtensionFilter("GeoTIFF", "tif", "tiff") });

            if (string.IsNullOrEmpty(savePath))
                return;

            _viewerScreen.ExportDem(savePath);
        }
    }
}