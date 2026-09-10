using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

/// <summary>
/// Batch-mode build entry point so the Windows player can be produced from
/// the command line (used by deployment docs / CI):
///   Unity -batchmode -quit -projectPath <proj> -buildTarget Win64 \
///         -executeMethod BuildScript.BuildWindows -logFile <path>
/// Outputs to <project>/build/DepthWizard/DepthWizard.exe (excluded from the
/// project via .gitignore; StreamingAssets is carried along automatically).
/// </summary>
public static class BuildScript
{
    public static void BuildWindows()
    {
        string[] scenes = { "Assets/Scenes/main.unity" };
        BuildReport report = BuildPipeline.BuildPlayer(
            scenes,
            "build/DepthWizard/DepthWizard.exe",
            BuildTarget.StandaloneWindows64,
            BuildOptions.None);
        if (report.summary.result != BuildResult.Succeeded)
            throw new BuildFailedException(
                $"Player build failed: {report.summary.result}");
        Debug.Log("BuildWindows completed.");
    }
}