import cv2
import numpy as np

def text_to_bits(text: str) -> str:
    bits = ''.join(format(ord(c), '08b') for c in text)
    return bits + "1111111111111110"  # delimitado a 16 bits

def psnr(original, modified) -> float:
    mse = np.mean((original.astype(np.float32) - modified.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def embed_dct_luma(img_bgr: np.ndarray, message: str, coeff_pos=(4, 3), q=10) -> np.ndarray:
    """
    Esteganografía DCT básica:
    - Convierte a YCrCb y trabaja en canal Y 
    - Divide en bloques 8x8
    - Aplica DCT por bloque
    - Modifica un coeficiente AC usando paridad con cuantización del paso q
    - Reconstruye con IDCT y vuelve a BGR
    """
    if img_bgr is None:
        raise ValueError("La imagen de entrada es de tipo None. Revisa cv2.imread().")

    bits = text_to_bits(message)

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32)

    h, w = Y.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    Yc = Y[:h8, :w8].copy()

    n_blocks = (h8 // 8) * (w8 // 8)
    if len(bits) > n_blocks:
        raise ValueError(f"Mensaje demasiado largo. Bits={len(bits)} > bloques={n_blocks}")

    u, v = coeff_pos
    if (u, v) == (0, 0):
        raise ValueError("No uses el coeficiente DC (0,0). Elige un AC, ej (4,3).")

    bit_idx = 0
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            if bit_idx >= len(bits):
                break

            block = Yc[i:i+8, j:j+8]
            dct_block = cv2.dct(block)

            c = dct_block[u, v]

            base = np.round(c / q) * q
            k = int(np.round(base / q)) 

            target_bit = int(bits[bit_idx])

            if (k % 2) != target_bit:
                cand1 = (k + 1) * q
                cand2 = (k - 1) * q
                base = cand1 if abs(cand1 - c) < abs(cand2 - c) else cand2

            dct_block[u, v] = base

            new_block = cv2.idct(dct_block)
            Yc[i:i+8, j:j+8] = new_block

            bit_idx += 1

        if bit_idx >= len(bits):
            break

    Y_out = Y.copy()
    Y_out[:h8, :w8] = np.clip(Yc, 0, 255)

    ycrcb_out = ycrcb.copy()
    ycrcb_out[:, :, 0] = Y_out.astype(np.uint8)

    out_bgr = cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)
    return out_bgr

# ---------- Main ----------
if __name__ == "__main__":
    img_path = "foto1.png"  # debe estar en la misma carpeta
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No pude abrir '{img_path}'. Pon la imagen en la misma carpeta del script.")

    # Parámetros DCT
    coeff_pos = (4, 3)  
    q_embed = 10         

    msg = "DCT stego + JPEG attack: Guillermo Dia 3."

    # 1) Generar estego DCT
    stego = embed_dct_luma(img, msg, coeff_pos=coeff_pos, q=q_embed)

    # 2) Guardar estego sin pérdidas (PNG)
    stego_png = f"stego_dct_pos{coeff_pos[0]}_{coeff_pos[1]}_q{q_embed}.png"
    cv2.imwrite(stego_png, stego)

    # 3) PSNR original vs estego
    p_stego = psnr(img, stego)
    print(f"Estego guardado: {stego_png}")
    print(f"PSNR (original vs estego): {p_stego:.2f} dB\n")

    # 4) Ataque JPEG + PSNR
    jpeg_qualities = [90, 70, 50]
    for qjpeg in jpeg_qualities:
        jpeg_name = f"attack_qjpeg{qjpeg}_from_stego.jpg"

        # Guardar como JPEG (ataque)
        cv2.imwrite(jpeg_name, stego, [int(cv2.IMWRITE_JPEG_QUALITY), qjpeg])

        # Recargar JPEG atacado
        attacked = cv2.imread(jpeg_name)
        if attacked is None:
            raise RuntimeError(f"No pude recargar el JPEG atacado: {jpeg_name}")

        p_attack = psnr(img, attacked)
        print(f"Ataque JPEG calidad {qjpeg}: PSNR (original vs atacada) = {p_attack:.2f} dB | archivo: {jpeg_name}")
