import torch
from thop import profile
from ultralytics import YOLO


def main():
    print("Loading model...")
    # Load your custom model
    model = YOLO("/home/shemin/PycharmProjects/leeyolo/runs/detect/lee_yolo/weights/best.pt")

    # Extract the raw PyTorch model and set it to evaluation mode
    pytorch_model = model.model.eval()

    # Create a "dummy" image tensor (1 image, 3 color channels, 640x640 pixels)
    # We put it on the exact same device (CPU or GPU) as the model
    dummy_input = torch.randn(1, 3, 640, 640).to(next(pytorch_model.parameters()).device)

    print("Forcing GFLOPs calculation through thop...")
    try:
        # Profile the model
        macs, params = profile(pytorch_model, inputs=(dummy_input,), verbose=False)

        # MACs to FLOPs is roughly a 2x conversion. Divide by 1 billion for GigaFLOPs.
        gflops = (macs * 2) / 1e9

        print("\n" + "=" * 40)
        print("✅ SUCCESS!")
        print(f"Total Parameters: {params:,}")
        print(f"Calculated GFLOPs: {gflops:.2f}")
        print("=" * 40 + "\n")

    except Exception as e:
        print(f"\n❌ The profiler failed to read your custom modules. Error: {e}")


if __name__ == "__main__":
    main()