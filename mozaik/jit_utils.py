import numpy as np
from numba import jit

# --- 1. GEOMETRIC KERNELS (Setup Phase) ---

@jit(nopython=True, cache=True, fastmath=True)
def fast_gabor(x1, y1, x2, y2, orientation, frequency, phase, size, aspect_ratio):
    # Safety guard for zero size
    if size < 1e-9:
        return 0.0
        
    dx = x1 - x2
    dy = y1 - y2
    cos_o = np.cos(orientation)
    sin_o = np.sin(orientation)
    X = dx * cos_o + dy * sin_o
    Y = -dx * sin_o + dy * cos_o
    
    X2 = X * X
    Y2 = Y * Y
    size2 = 2 * (size * size)
    ar2 = aspect_ratio * aspect_ratio
    
    return np.exp(-(X2 + Y2 * ar2) / size2) * np.cos(2 * np.pi * X * frequency + phase)

@jit(nopython=True, cache=True, fastmath=True)
def fast_gauss(x1, y1, x2, y2, orientation, size, aspect_ratio):
    # Safety guard
    if size < 1e-9:
        return 0.0

    dx = x1 - x2
    dy = y1 - y2
    cos_o = np.cos(orientation)
    sin_o = np.sin(orientation)
    X = dx * cos_o + dy * sin_o
    Y = -dx * sin_o + dy * cos_o
    
    return np.exp(-(X*X + Y*Y*(aspect_ratio*aspect_ratio)) / (2*(size*size)))

# --- 2. RECEPTIVE FIELD UPDATE (Simulation Phase) ---

@jit(nopython=True, cache=True, fastmath=True)
def fast_stf_view_update(view_array, kernel_contrast, background_lum, 
                         contrast_resp, luminance_resp, 
                         kernel_luminance, mean_val, 
                         current_idx, update_factor, duration):
    flat_view = view_array.ravel()
    
    # Avoid division by zero
    if background_lum < 1e-9:
        scaled_input = flat_view # fallback or 0 depending on logic, usually safe to leave or set 0
    else:
        scaled_input = flat_view / background_lum
    
    # Dot product
    contrast_tc = np.dot(kernel_contrast, scaled_input)
    luminance_tc = kernel_luminance * mean_val
    
    # Accumulate loop
    limit = min(duration, contrast_resp.shape[0] - current_idx)
    for j in range(update_factor):
        target = current_idx + j
        if target + limit <= contrast_resp.shape[0]:
             contrast_resp[target : target + limit] += contrast_tc[:limit]
             luminance_resp[target : target + limit] += luminance_tc[:limit]

# --- 3. FULL INTEGRAL CALCULATION (Connectivity Phase) ---

@jit(nopython=True, cache=True, fastmath=True)
def fast_integral_vectorized(K1, wx1, wy1, px1, py1, gor1, freq1, sor1, ph1,
                             K2, wx2, wy2, px2, py2, gor2, freq2, sor2, ph2):
    n = len(K1)
    result = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        # 1. Calculate Omegas
        # Omega 1
        co1 = np.cos(gor1[i])
        so1 = np.sin(gor1[i])
        wx1_sq = wx1[i]**2
        wy1_sq = wy1[i]**2
        O1a = wx1_sq * co1**2 + wy1_sq * so1**2
        O1b = co1 * so1 * (wx1_sq - wy1_sq)
        O1d = wx1_sq * so1**2 + wy1_sq * co1**2
        
        # Omega 2
        co2 = np.cos(gor2[i])
        so2 = np.sin(gor2[i])
        wx2_sq = wx2[i]**2
        wy2_sq = wy2[i]**2
        O2a = wx2_sq * co2**2 + wy2_sq * so2**2
        O2b = co2 * so2 * (wx2_sq - wy2_sq)
        O2d = wx2_sq * so2**2 + wy2_sq * co2**2

        # 2. Sum Omega (OS)
        OSa, OSb, OSd = O1a + O2a, O1b + O2b, O1d + O2d
        
        # 3. Inverse Omega Sum (OSInv)
        # Determinant of 2x2 matrix
        det = OSa * OSd - OSb**2
        
        # --- NUMERICAL STABILITY FIX ---
        # If determinant is 0 or negative (due to float precision), 
        # the integral is invalid/divergent or 0.
        if det <= 1e-15:
            result[i] = 0.0
            continue
        # -------------------------------

        invDet = 1.0 / det
        OSInva =  OSd * invDet
        OSInvb = -OSb * invDet
        OSInvd =  OSa * invDet

        # 4. Matrix Math Unrolled
        # _x1 = O1 * pos1
        _x1_0 = O1a * px1[i] + O1b * py1[i]
        _x1_1 = O1b * px1[i] + O1d * py1[i]
        
        # _x2 = O2 * pos2
        _x2_0 = O2a * px2[i] + O2b * py2[i]
        _x2_1 = O2b * px2[i] + O2d * py2[i]
        
        # xs = OSInv * (_x1 + _x2)
        _sum_x = _x1_0 + _x2_0
        _sum_y = _x1_1 + _x2_1
        xs0 = OSInva * _sum_x + OSInvb * _sum_y
        xs1 = OSInvb * _sum_x + OSInvd * _sum_y
        
        # 5. K_s
        term1 = px1[i] * (O1a * px1[i] + O1b * py1[i]) + py1[i] * (O1b * px1[i] + O1d * py1[i])
        term2 = px2[i] * (O2a * px2[i] + O2b * py2[i]) + py2[i] * (O2b * px2[i] + O2d * py2[i])
        term3 = xs0 * (OSa * xs0 + OSb * xs1) + xs1 * (OSb * xs0 + OSd * xs1)
        
        # Check for overflow in exponent
        exp_arg = -np.pi * (term1 + term2 - term3)
        if exp_arg < -700: # approx min for float64 exp
            Ks = 0.0
        else:
            Ks = K1[i] * K2[i] * np.exp(exp_arg)
        
        # 6. Integrals
        ux1 = freq1[i] * np.cos(sor1[i])
        uy1 = freq1[i] * np.sin(sor1[i])
        ux2 = freq2[i] * np.cos(sor2[i])
        uy2 = freq2[i] * np.sin(sor2[i])
        
        # Helper inline for Integral 1
        udx = ux1 - ux2
        udy = uy1 - uy2
        quad_u = udx * (OSInva * udx + OSInvb * udy) + udy * (OSInvb * udx + OSInvd * udy)
        dot_u_xs = udx * xs0 + udy * xs1
        
        # Check for valid math
        pre_factor = Ks / np.sqrt(det) # safe because det > 1e-15
        val1 = pre_factor * np.exp(-np.pi * quad_u) * np.cos(2 * np.pi * dot_u_xs + (ph1[i] - ph2[i]))

        # Helper inline for Integral 2
        usx = ux1 + ux2
        usy = uy1 + uy2
        quad_u2 = usx * (OSInva * usx + OSInvb * usy) + usy * (OSInvb * usx + OSInvd * usy)
        dot_u2_xs = usx * xs0 + usy * xs1
        val2 = pre_factor * np.exp(-np.pi * quad_u2) * np.cos(2 * np.pi * dot_u2_xs + (ph1[i] + ph2[i]))

        # Ensure no NaNs leaked through
        res = 0.5 * (val1 + val2)
        if np.isnan(res):
            result[i] = 0.0
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        else:
            result[i] = res
        
    return result

@jit(nopython=True, cache=True, fastmath=True)
def fast_overlay(scene, obj_view):
    """
    Overlays obj_view onto scene in-place, respecting transparency.
    Assuming TRANSPARENT is -1.
    """
    h, w = scene.shape
    # Flatten loop for speed
    scene_flat = scene.ravel()
    view_flat = obj_view.ravel()
    
    for i in range(len(scene_flat)):
        val = view_flat[i]
        # If the new object is not transparent (val > -1 + epsilon), overwrite the scene
        # We assume TRANSPARENT is exactly -1, but let's be safe with > -0.9 if it's float
        if val > -0.99: 
            scene_flat[i] = val
    
    return scene

@jit(nopython=True, cache=True, fastmath=True)
def fast_replace_transparent(scene, background_lum):
    """
    Replaces remaining transparent pixels with background luminance.
    """
    scene_flat = scene.ravel()
    for i in range(len(scene_flat)):
        if scene_flat[i] <= -0.99:
            scene_flat[i] = background_lum
    return scene

# Append to mozaik/jit_utils.py

@jit(nopython=True, cache=True, fastmath=True)
def fast_extract_patch(img, view,
                       img_rel_left, img_rel_width, img_rel_top, img_rel_height,
                       view_rel_left, view_rel_width, view_rel_top, view_rel_height):
    """
    Extracts a patch from 'img' and places it into 'view' based on relative coordinates.
    This replaces the heavy slicing logic in VisualStimulus.display.
    """
    img_h, img_w = img.shape
    view_h, view_w = view.shape
    
    # --- 1. Calculate Source (Image) Coordinates ---
    j_start = int(round(img_rel_left * img_w))
    delta_j = int(round(img_rel_width * img_w))
    
    # Note: Original code used "img.shape[0] - ..." for top, implying inverted Y axis?
    # Original: i_start = img.shape[0] - numpy.round(img_relative_top * img.shape[0]).astype(int)
    # We replicate exactly:
    i_inv_start = int(round(img_rel_top * img_h))
    i_start = img_h - i_inv_start
    
    delta_i = int(round(img_rel_height * img_h))
    
    # --- 2. Calculate Target (View) Coordinates ---
    l_start = int(round(view_rel_left * view_w))
    delta_l = int(round(view_rel_width * view_w))
    
    k_inv_start = int(round(view_rel_top * view_h))
    k_start = view_h - k_inv_start
    
    delta_k = int(round(view_rel_height * view_h))
    
    # --- 3. Consistency Fix (from original code) ---
    if abs(delta_j - delta_l) == 1:
        m = min(delta_j, delta_l)
        delta_j = m
        delta_l = m
        
    if abs(delta_i - delta_k) == 1:
        m = min(delta_i, delta_k)
        delta_i = m
        delta_k = m
    
    # --- 4. Bounds Calculation ---
    i_stop = i_start + delta_i
    j_stop = j_start + delta_j
    k_stop = k_start + delta_k
    l_stop = l_start + delta_l
    
    # --- 5. Validating Bounds (Safety) ---
    # Ensure we don't slice out of bounds, which crashes Numba
    if i_start < 0: i_start = 0
    if j_start < 0: j_start = 0
    if k_start < 0: k_start = 0
    if l_start < 0: l_start = 0
    
    if i_stop > img_h: i_stop = img_h
    if j_stop > img_w: j_stop = img_w
    if k_stop > view_h: k_stop = view_h
    if l_stop > view_w: l_stop = view_w
    
    # --- 6. The Copy Loop (Manual 2D copy is faster than slicing in Numba often) ---
    # We iterate over the target area
    
    # Effective width/height to copy
    h_copy = min(i_stop - i_start, k_stop - k_start)
    w_copy = min(j_stop - j_start, l_stop - l_start)
    
    if h_copy > 0 and w_copy > 0:
        for r in range(h_copy):
            for c in range(w_copy):
                view[k_start + r, l_start + c] = img[i_start + r, j_start + c]
                
    return view

@jit(nopython=True, cache=True, fastmath=True)
def fast_resize(img, zoom):
    """
    Fast Bilinear Interpolation for 2D arrays.
    Replaces scipy.ndimage.zoom(img, zoom, order=3).
    """
    h, w = img.shape
    # Calculate new dimensions
    new_h = int(h * zoom + 0.5)
    new_w = int(w * zoom + 0.5)
    
    output = np.empty((new_h, new_w), dtype=img.dtype)
    
    for r in range(new_h):
        for c in range(new_w):
            # Map target pixel back to source coordinates
            src_r = r / zoom
            src_c = c / zoom
            
            # Floor coordinates
            r0 = int(src_r)
            c0 = int(src_c)
            
            # Clamp to boundaries
            if r0 >= h - 1: r0 = h - 2
            if c0 >= w - 1: c0 = w - 2
            
            r1 = r0 + 1
            c1 = c0 + 1
            
            # Interpolation weights
            dr = src_r - r0
            dc = src_c - c0
            
            # Bilinear interpolation
            # f(x,y) = (1-dx)(1-dy)f00 + (1-dx)dyf01 + dx(1-dy)f10 + dxdyf11
            # Note: rows=y (dr), cols=x (dc)
            val = (1 - dr) * (1 - dc) * img[r0, c0] + \
                  (1 - dr) * dc       * img[r0, c1] + \
                  dr       * (1 - dc) * img[r1, c0] + \
                  dr       * dc       * img[r1, c1]
            
            output[r, c] = val
            
    return output