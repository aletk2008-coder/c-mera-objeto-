import cv2
from ultralytics import YOLO

def iniciar_scanner():
    """
    Inicializa a captura de vídeo da webcam local e executa a detecção 
    de objetos com o YOLO em tempo real via OpenCV.
    """
    # Carrega o modelo YOLO v8 nano
    model = YOLO("yolov8n.pt")

    # Inicializa a captura da webcam padrão (índice 0)
    cap = cv2.VideoCapture(0)

    # Verifica se a câmera foi aberta corretamente
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return

    print("Scanner iniciado. Pressione 'q' para sair.")

    while True:
        # Captura o frame atual da câmera
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar o frame da imagem.")
            break

        # Executa a inferência do YOLO no frame capturado
        results = model(frame)

        # Plota os resultados (caixas delimitadoras e rótulos) no frame original
        annotated_frame = results[0].plot()

        # Exibe o frame processado em uma janela nativa do OpenCV
        cv2.imshow("Scanner com YOLO", annotated_frame)

        # Interrompe o loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos da câmera e fecha todas as janelas abertas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_scanner()