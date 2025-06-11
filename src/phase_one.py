import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter


def get_initial_point(noisy_image, method='noisy', **kwargs):
    """
    Get initial point for TV denoising optimization

    Parameters:
    -----------
    noisy_image : np.ndarray
        The noisy input image
    method : str
        Initialization method: 'noisy', 'gaussian', 'bilateral', 'median', 'mean', 'tv_l2'
    **kwargs : dict
        Method-specific parameters

    Returns:
    --------
    x0 : np.ndarray
        Initial point for optimization
    """

    if method == 'noisy':
        # Simplest: start with noisy image itself
        return noisy_image.copy()

    elif method == 'gaussian':
        # Gaussian smoothing - good general purpose initialization
        sigma = kwargs.get('sigma', 1.0)
        return gaussian_filter(noisy_image, sigma=sigma)

    elif method == 'bilateral':
        # Bilateral filter - preserves edges while smoothing
        try:
            import cv2
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)

            # Normalize to 0-255 for cv2
            img_norm = ((noisy_image - noisy_image.min()) /
                        (noisy_image.max() - noisy_image.min()) * 255).astype(np.uint8)

            filtered = cv2.bilateralFilter(img_norm, d, sigma_color, sigma_space)

            # Scale back to original range
            filtered = filtered.astype(np.float64) / 255.0
            filtered = filtered * (noisy_image.max() - noisy_image.min()) + noisy_image.min()

            return filtered

        except ImportError:
            print("OpenCV not available, falling back to Gaussian filter")
            return gaussian_filter(noisy_image, sigma=1.0)

    elif method == 'median':
        # Median filter - good for salt-and-pepper noise
        size = kwargs.get('size', 3)
        return ndimage.median_filter(noisy_image, size=size)

    elif method == 'mean':
        # Simple mean filter
        size = kwargs.get('size', 3)
        kernel = np.ones((size, size)) / (size * size)
        return ndimage.convolve(noisy_image, kernel, mode='reflect')

    elif method == 'tv_l2':
        # Quick TV-L2 denoising as initialization
        # This is a simplified version that's faster than full TV
        return tv_l2_init(noisy_image, **kwargs)

    elif method == 'adaptive':
        # Adaptive method based on noise characteristics
        return adaptive_init(noisy_image, **kwargs)

    else:
        raise ValueError(f"Unknown initialization method: {method}")


def tv_l2_init(noisy_image, lambda_tv=0.1, lambda_l2=1.0, max_iter=50):
    """
    Quick TV-L2 denoising for initialization
    Uses a simpler gradient descent approach
    """
    x = noisy_image.copy()
    m, n = x.shape

    for _ in range(max_iter):
        # Compute TV gradient (simplified)
        Dx = np.diff(x, axis=0, append=x[-1:, :])
        Dy = np.diff(x, axis=1, append=x[:, -1:])

        epsilon = 1e-6
        norm = np.sqrt(Dx ** 2 + Dy ** 2) + epsilon

        # TV gradient
        tv_grad = np.zeros_like(x)
        tv_grad += Dx / norm
        tv_grad += Dy / norm
        tv_grad[1:, :] -= Dx[:-1, :] / norm[:-1, :]
        tv_grad[:, 1:] -= Dy[:, :-1] / norm[:, :-1]

        # L2 gradient (data fidelity)
        l2_grad = x - noisy_image

        # Gradient step
        total_grad = lambda_tv * tv_grad + lambda_l2 * l2_grad
        step_size = 0.01
        x -= step_size * total_grad

    return x


def adaptive_init(noisy_image, noise_std=None):
    """
    Adaptive initialization based on estimated noise characteristics
    """
    if noise_std is None:
        # Estimate noise using robust median estimator
        noise_std = estimate_noise_std(noisy_image)

    # Choose method based on noise level
    if noise_std < 0.02:  # Low noise
        return gaussian_filter(noisy_image, sigma=0.5)
    elif noise_std < 0.1:  # Medium noise
        return gaussian_filter(noisy_image, sigma=1.0)
    else:  # High noise
        return tv_l2_init(noisy_image, lambda_tv=0.2, max_iter=30)


def estimate_noise_std(image):
    """
    Estimate noise standard deviation using robust method
    Based on median absolute deviation of Laplacian
    """
    # Laplacian kernel
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])

    # Apply Laplacian
    laplacian = ndimage.convolve(image, laplacian_kernel, mode='reflect')

    # Robust noise estimation
    sigma = np.median(np.abs(laplacian)) / 0.6745

    return sigma


def validate_initialization(x0, noisy_image, tv_evaluator):
    """
    Validate and potentially improve the initialization
    """
    # Check if initialization is reasonable
    tv_val, grad, _ = tv_evaluator.eval(x0)
    grad_norm = np.linalg.norm(grad)

    print(f"Initial TV value: {tv_val:.6f}")
    print(f"Initial gradient norm: {grad_norm:.6f}")
    print(f"Data fidelity: {np.linalg.norm(x0 - noisy_image):.6f}")

    # If gradient is too large, the initialization might be poor
    if grad_norm > 1000:
        print("Warning: Large initial gradient, trying smoother initialization")
        x0_smooth = gaussian_filter(x0, sigma=2.0)
        return x0_smooth

    return x0


# Example usage function
def get_best_initialization(noisy_image, tv_evaluator=None):
    """
    Try multiple initialization methods and pick the best one
    """
    methods = [
        ('gaussian', {'sigma': 0.5}),
        ('gaussian', {'sigma': 1.0}),
        ('gaussian', {'sigma': 2.0}),
        ('tv_l2', {'lambda_tv': 0.1, 'max_iter': 20}),
        ('adaptive', {})
    ]

    best_x0 = None
    best_score = float('inf')

    for method, params in methods:
        try:
            x0 = get_initial_point(noisy_image, method=method, **params)

            # Score based on TV value and data fidelity
            if tv_evaluator is not None:
                tv_val, _, _ = tv_evaluator.eval(x0)
                data_fidelity = np.linalg.norm(x0 - noisy_image) ** 2
                score = tv_val + 0.5 * data_fidelity  # Simple weighted combination
            else:
                # Fallback scoring without TV evaluator
                score = np.var(x0) + np.linalg.norm(x0 - noisy_image) ** 2

            print(f"Method {method}: score = {score:.6f}")

            if score < best_score:
                best_score = score
                best_x0 = x0
                best_method = method

        except Exception as e:
            print(f"Method {method} failed: {e}")
            continue

    print(f"Best initialization method: {best_method}")
    return best_x0


# Practical example
def example_usage():
    """
    Example of how to use these initialization methods
    """
    # Create synthetic noisy image
    np.random.seed(42)
    clean_image = np.zeros((50, 50))
    clean_image[20:30, 20:30] = 1.0  # Square
    noisy_image = clean_image + 0.1 * np.random.randn(50, 50)

    # Try different initializations
    print("=== Initialization Comparison ===")

    methods_to_try = [
        ('noisy', {}),
        ('gaussian', {'sigma': 1.0}),
        ('median', {'size': 3}),
        ('tv_l2', {'lambda_tv': 0.1, 'max_iter': 20}),
        ('adaptive', {})
    ]

    for method, params in methods_to_try:
        x0 = get_initial_point(noisy_image, method=method, **params)
        mse = np.mean((x0 - clean_image) ** 2)
        print(f"{method:10s}: MSE = {mse:.6f}")



if __name__ == "__main__":
    example_usage()