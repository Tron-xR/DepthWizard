using UnityEngine;

namespace DepthWizard.Camera
{
    public class FreeFlyCamera : MonoBehaviour
    {
        [Header("Movement")]
        [SerializeField] private float _moveSpeed = 20f;
        [SerializeField] private float _boostMultiplier = 3f;
        [SerializeField] private float _scrollSpeed = 5f;

        [Header("Look")]
        [SerializeField] private float _mouseSensitivity = 3f;
        [SerializeField] private float _pitchClamp = 89f;

        private float _yaw;
        private float _pitch;
        private bool _cursorLocked;

        private void Start()
        {
            SyncState();
        }

        public void SyncState()
        {
            Vector3 euler = transform.eulerAngles;
            _yaw = euler.y;
            _pitch = euler.x;
        }

        private void Update()
        {
            HandleCursorLock();
            HandleRotation();
            HandleMovement();
        }

        private void HandleCursorLock()
        {
            if (Input.GetMouseButtonDown(1))
            {
                Cursor.lockState = CursorLockMode.Locked;
                Cursor.visible = false;
                _cursorLocked = true;
            }
            else if (Input.GetMouseButtonUp(1))
            {
                Cursor.lockState = CursorLockMode.None;
                Cursor.visible = true;
                _cursorLocked = false;
            }
        }

        private void HandleRotation()
        {
            if (!_cursorLocked) return;

            float mx = Input.GetAxis("Mouse X") * _mouseSensitivity;
            float my = Input.GetAxis("Mouse Y") * _mouseSensitivity;

            _yaw += mx;
            _pitch -= my;
            _pitch = Mathf.Clamp(_pitch, -_pitchClamp, _pitchClamp);

            transform.rotation = Quaternion.Euler(_pitch, _yaw, 0f);
        }

        private void HandleMovement()
        {
            float speed = _moveSpeed;
            if (Input.GetKey(KeyCode.LeftShift))
                speed *= _boostMultiplier;

            // Scroll wheel adjusts base speed (not applied directly to position)
            float scroll = Input.GetAxis("Mouse ScrollWheel");
            if (Mathf.Abs(scroll) > 0.001f)
                _moveSpeed = Mathf.Clamp(_moveSpeed + scroll * _scrollSpeed * 10f, 1f, 500f);

            Vector3 dir = Vector3.zero;

            if (Input.GetKey(KeyCode.W)) dir += transform.forward;
            if (Input.GetKey(KeyCode.S)) dir -= transform.forward;
            if (Input.GetKey(KeyCode.A)) dir -= transform.right;
            if (Input.GetKey(KeyCode.D)) dir += transform.right;
            if (Input.GetKey(KeyCode.E)) dir += Vector3.up;
            if (Input.GetKey(KeyCode.Q)) dir -= Vector3.up;

            if (dir.sqrMagnitude > 0f)
                dir.Normalize();

            transform.position += dir * speed * Time.deltaTime;
        }
    }
}
