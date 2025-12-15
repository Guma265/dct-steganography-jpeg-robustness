Este repositorio presenta una implementación mínima y reproducible de esteganografía de imágenes en el dominio DCT, así como la evaluación de su robustez frente a ataques de compresión JPEG.

Metodología
Espacio de color: YCrCb (canal de luminancia Y)
Transformada: DCT 2D aplicada sobre bloques de 8×8
Embebido: modulación por paridad de un coeficiente AC de frecuencia media
Fuerza de embebido: paso de cuantización q
Métrica de evaluación: PSNR

Experimentos
Embebido en el dominio DCT con control explícito de la fuerza de embebido (q)
Ataques de compresión JPEG (calidades = 90, 70, 50)
Evaluación de la fidelidad visual mediante PSNR

Resultados (ejemplo realizado)
PSNR (original vs estego): ~54.6 dB
PSNR tras JPEG 90: ~33.9 dB
PSNR tras JPEG 70: ~31.2 dB
PSNR tras JPEG 50: ~29.6 dB

Estructura del repositorio
src/: código fuente
results/: imágenes estego y atacadas generadas

Trabajo futuro
Extracción del mensaje y evaluación de la tasa de error de bits (BER)
Comparación con métodos basados en LSB
Experimentos de esteganálisis
