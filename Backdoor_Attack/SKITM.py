class SparseKeyframeSelector:
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.9):
        self.alpha = alpha
        self.beta = 1 - alpha
        self.gamma = gamma

    def _preprocess_images(self, frames, ref_frame):
        """Preprocess all images into numpy arrays and grayscale images"""
        # Convert the reference frame
        ref_np = np.array(ref_frame)
        ref_gray = cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY)

        # Convert all frames in batch
        frame_nps = [cv2.resize(np.array(f), (224, 224)) for f in frames]
        frame_grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frame_nps]

        return ref_np, ref_gray, frame_nps, frame_grays

    def _compute_pixel_diff(self, frame_np, ref_np):
        """Optimized pixel-level difference computation (inputs are already numpy arrays)"""
        diff = np.abs(frame_np.astype(np.float32) - ref_np.astype(np.float32))
        # Normalize each channel independently
        min_vals = diff.min(axis=(0, 1), keepdims=True)
        max_vals = diff.max(axis=(0, 1), keepdims=True)
        norm_diff = (diff - min_vals) / (max_vals - min_vals + 1e-8)
        return np.mean(np.linalg.norm(norm_diff, axis=2))

    def _compute_flow_energy(self, ref_gray, frame_gray):
        """Optimized optical flow energy computation (inputs are preprocessed grayscale images)"""
        flow = cv2.calcOpticalFlowFarneback(
            ref_gray, frame_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX).mean()

    def select_keyframes(self, frames, ref_frame):
        """
        Optimized version supporting PIL Image input.
        Input:
            frames: list of PIL.Image [T]
            ref_frame: PIL.Image
        Output:
            mask: ndarray [T], boolean type
        """
        T = len(frames)
        if T == 0:
            return np.zeros(0, dtype=bool)

        # Step 1: Batch preprocess images
        ref_np, ref_gray, frame_nps, frame_grays = self._preprocess_images(frames, ref_frame)

        # Step 2: Compute all metrics in parallel
        with ThreadPoolExecutor() as executor:
            # Compute pixel-level differences in parallel
            pixel_diffs = list(executor.map(
                lambda f: self._compute_pixel_diff(f, ref_np),
                frame_nps
            ))

            # Compute optical flow energy in parallel
            flow_energies = list(executor.map(
                lambda g: self._compute_flow_energy(ref_gray, g),
                frame_grays
            ))

        # Convert to numpy arrays
        pixel_diffs = np.array(pixel_diffs)
        flow_energies = np.array(flow_energies)

        # Step 3: Combined scoring (vectorized computation)
        scores = self.alpha * pixel_diffs
        scores[1:-1] += self.beta * 0.5 * (flow_energies[:-2] + flow_energies[1:-1])
        scores[0] += self.beta * flow_energies[0]
        if T > 1:
            scores[-1] += self.beta * flow_energies[-1]

        # Step 4: Temporal smoothing (vectorized)
        smoothed = np.convolve(scores, [1 / 3, 1 / 3, 1 / 3], mode='same')
        smoothed[0] = np.mean(scores[:2])
        smoothed[-1] = np.mean(scores[-2:])

        # Step 5: Dynamic threshold
        threshold = self.gamma * np.max(smoothed)
        mask = smoothed >= threshold
        return mask
