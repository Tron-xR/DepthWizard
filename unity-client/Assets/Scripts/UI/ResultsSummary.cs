using UnityEngine;
using TMPro;
using DepthWizard.Core;
using DepthWizard.Terrain;

namespace DepthWizard.UI
{
    /// <summary>
    /// Formats the already-fetched validation metrics (Pearson r, MAE, RMSE,
    /// Bias, Prediction std, Reference std, Calibration scale/offset) plus the
    /// active backend slug and the uncalibrated-model polarity trio (raw
    /// correlation, inverted flag, polarity reason) into a single multi-line
    /// text block and swaps it in place of the terrain model.
    ///
    /// This script creates no button/panel layout - wire it manually in the
    /// Inspector: drag your centered TMP Text into <see cref="resultsText"/>, the
    /// ViewerScreen whose <see cref="ViewerScreen.LastValidation"/> holds the
    /// fetched /validate result into <see cref="viewerScreen"/>, and the terrain's
    /// MeshGenerator into <see cref="meshGenerator"/>. Then bind your button's
    /// onClick to <see cref="DisplayLastLoaded"/> (no arguments) to toggle the
    /// block on/off in place of the model, or to <see cref="ToggleResultsView(bool)"/>
    /// to show/hide without refetching.
    /// </summary>
    public class ResultsSummary : MonoBehaviour
    {
        [Header("Output")]
        [Tooltip("Single TMP Text shown centered in place of the terrain model.")]
        public TMP_Text resultsText;

        [Header("Data source")]
        [Tooltip("ViewerScreen that owns the fetched /validate result (LastValidation).")]
        public ViewerScreen viewerScreen;

        [Header("Terrain")]
        [Tooltip("Terrain model object hidden while results are shown; the same MeshGenerator that builds the mesh.")]
        public MeshGenerator meshGenerator;

        [Header("Legacy readout")]
        [Tooltip("Older always-on /validate readout (RMSE/MAE/Correlation block) owned by the viewer screen's own text. Hidden while the results block is shown so the two readouts never overlap.")]
        public TMP_Text validationReadout;

        /// <summary>
        /// Format an already-fetched /validate result into the results block and
        /// assign it to <see cref="resultsText"/> (text object shown). Values that
        /// the server did not return render as "--" instead of a fake 0.
        /// </summary>
        public void ShowResults(ValidationResponse result)
        {
            if (resultsText == null)
            {
                Debug.LogError("ResultsSummary.ShowResults: resultsText is not assigned in the Inspector.");
                return;
            }
            if (result == null)
            {
                resultsText.text = "";
                resultsText.gameObject.SetActive(false);
                return;
            }
            resultsText.text = FormatResults(result);
            resultsText.gameObject.SetActive(true);
        }

        /// <summary>True while the results text object is active - i.e. the results
        /// block is the current view, not the terrain model. Derived straight from
        /// the GameObject so it can never drift from what is on screen.</summary>
        public bool IsShowingResults => resultsText != null && resultsText.gameObject.activeSelf;

        /// <summary>
        /// Show/hide the results block in place of the terrain: when showResults is
        /// true the model's GameObject is deactivated (the same SetActive mechanism
        /// the DEM view / display-mode toggles use), the DEM overlay is hidden
        /// (results view is exclusive; the DEM toggle stays off until the user
        /// turns it back on), the legacy always-on validate readout is hidden so
        /// the two readouts never overlap, and the results text object is revealed;
        /// false restores the model and readout and hides the text without touching
        /// the DEM overlay's own state.
        /// </summary>
        public void ToggleResultsView(bool showResults)
        {
            if (showResults && viewerScreen != null)
                viewerScreen.SetDemViewShown(false);
            if (meshGenerator != null)
                meshGenerator.gameObject.SetActive(!showResults);
            if (validationReadout != null)
                validationReadout.gameObject.SetActive(!showResults);
            if (resultsText != null)
                resultsText.gameObject.SetActive(showResults);
        }

        /// <summary>
        /// One-call entry point for a button's onClick; toggles the results block
        /// on and off. When the block is currently shown it hides it and restores
        /// the terrain model (no re-fetch). Otherwise it shows the most recent
        /// /validate result (from <see cref="viewerScreen"/>.LastValidation) as a
        /// formatted block and hides the terrain model. No-op (with a log) while
        /// no validation has been fetched - run the existing Validate flow first.
        /// </summary>
        public void DisplayLastLoaded()
        {
            if (IsShowingResults)
            {
                ToggleResultsView(false);
                return;
            }
            if (viewerScreen == null)
            {
                Debug.LogError("ResultsSummary.DisplayLastLoaded: viewerScreen is not assigned in the Inspector.");
                return;
            }
            ValidationResponse last = viewerScreen.LastValidation;
            if (last == null)
            {
                Debug.LogWarning("ResultsSummary: no validation result loaded yet. Run Validate first.");
                ShowResults(last);
                return;
            }
            ShowResults(last);
            ToggleResultsView(true);
        }

        /// <summary>One metric row: padded label, then the value or "--" when the
        /// server did not send the field. Pad so monospaced blocks line up the
        /// values in one column.</summary>
        private static string Row(string label, bool has, float value, string format, string unit)
        {
            string v = has ? value.ToString(format) : "--";
            return $"{label.PadRight(18)}: {v}{(has && unit.Length > 0 ? " " + unit : "")}";
        }

        private static string TextRow(string label, bool has, string value)
        {
            return $"{label.PadRight(18)}: {(has && !string.IsNullOrEmpty(value) ? value : "--")}";
        }

        private static string FormatResults(ValidationResponse r)
        {
            // Formatting matches the existing validate readout (Correlation F3,
            // RMSE/MAE F2 m) extended to the additive fields: bias and stds keep
            // 3 decimals per the results-block spec, RMSE/MAE stay 2.
            return string.Join("\n",
                TextRow("Backend", r.hasBackend, r.backend),
                Row("Pearson r", r.hasCorrelation, r.correlation, "F3", ""),
                Row("MAE", true, r.mae, "F2", "m"),
                Row("RMSE", true, r.rmse, "F2", "m"),
                Row("Bias", r.hasBias, r.bias, "F3", "m"),
                Row("Prediction std", r.hasPredictionStd, r.prediction_std, "F3", "m"),
                Row("Reference std", r.hasReferenceStd, r.reference_std, "F3", "m"),
                Row("Calibration scale", r.hasCalibrationScale, r.calibration_scale, "F3", ""),
                Row("Calibration offset", r.hasCalibrationOffset, r.calibration_offset, "F2", "m"),
                // Uncalibrated-model polarity trio: the calibrated Pearson r above
                // is flipped positive when the fit's scale is negative, so it cannot
                // distinguish a genuine +r from a masked negative raw correlation.
                // raw_correlation_signed keeps its sign; Inverted + reason surface
                // the server's polarity verdict when the flag was sent.
                Row("Raw corr", r.hasRawCorrelation, r.raw_correlation_signed, "F3", ""),
                TextRow("Inverted", r.hasPolarityInverted, r.polarity_inverted ? "yes" : "no"),
                TextRow("Polarity reason", r.hasPolarityReason, r.polarity_reason));
        }
    }
}