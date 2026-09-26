using System.Collections;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// Recreates this Uiverse button's hover behavior in Unity UI:
///
///   button {
///     background-color: #f3f7fe;
///     color: #3b82f6;
///     border-radius: 8px;
///     width: 100px; height: 45px;
///     transition: 0.3s;
///   }
///   button:hover {
///     background-color: #3b82f6;
///     box-shadow: 0 0 0 5px #3b83f65f;
///     color: #fff;
///   }
///
/// SETUP:
/// 1. Put this on the button root (the GameObject with the Image + Button).
/// 2. Assign "background" to that same Image (9-sliced rounded-rect sprite, 8px radius).
/// 3. Assign "label" to the child TextMeshProUGUI showing the button text.
/// 4. Create a second child GameObject BEHIND the button (lower sibling index),
///    same rounded-rect sprite, sized slightly larger than the button (e.g. +10px
///    on each side to mimic the 5px CSS shadow spread). Set its Image alpha to 0
///    at rest. Assign it to "glow".
/// 5. Tune restBg/hoverBg/restText/hoverText/glowColor in the Inspector if you
///    want different colors for a dark-HUD palette instead of the light defaults.
/// </summary>
[RequireComponent(typeof(Image))]
public class MinimalButtonHover : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler
{
    [Header("References")]
    public Image background;      // the button's own Image
    public TMP_Text label;         // button text
    public Image glow;             // larger rounded-rect behind the button, alpha 0 at rest

    [Header("Colors")]
    public Color restBg = new Color(0.953f, 0.969f, 0.996f);   // #f3f7fe
    public Color hoverBg = new Color(0.231f, 0.510f, 0.965f);  // #3b82f6
    public Color restText = new Color(0.231f, 0.510f, 0.965f); // #3b82f6
    public Color hoverText = Color.white;
    [Range(0f, 1f)] public float glowAlpha = 0.37f;            // ~ #3b83f65f alpha

    [Header("Timing")]
    public float duration = 0.3f; // matches CSS "transition: 0.3s"

    Coroutine _running;

    void Reset()
    {
        background = GetComponent<Image>();
    }

    public void OnPointerEnter(PointerEventData eventData) => Animate(true);
    public void OnPointerExit(PointerEventData eventData) => Animate(false);

    void Animate(bool hovering)
    {
        if (_running != null) StopCoroutine(_running);
        _running = StartCoroutine(AnimateRoutine(hovering));
    }

    IEnumerator AnimateRoutine(bool hovering)
    {
        Color bgFrom = background.color;
        Color bgTo = hovering ? hoverBg : restBg;

        Color textFrom = label != null ? label.color : Color.white;
        Color textTo = hovering ? hoverText : restText;

        float glowFrom = glow != null ? glow.color.a : 0f;
        float glowTo = hovering ? glowAlpha : 0f;

        float t = 0f;
        while (t < duration)
        {
            t += Time.deltaTime;
            float k = Mathf.Clamp01(t / duration);
            // ease-out, roughly matching CSS default easing
            k = 1f - Mathf.Pow(1f - k, 2f);

            background.color = Color.Lerp(bgFrom, bgTo, k);
            if (label != null) label.color = Color.Lerp(textFrom, textTo, k);
            if (glow != null)
            {
                Color c = glow.color;
                c.a = Mathf.Lerp(glowFrom, glowTo, k);
                glow.color = c;
            }
            yield return null;
        }

        background.color = bgTo;
        if (label != null) label.color = textTo;
        if (glow != null)
        {
            Color c = glow.color;
            c.a = glowTo;
            glow.color = c;
        }
    }
}
