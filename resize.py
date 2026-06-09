import cv2
import os

def batch_resize(input_dir, output_dir, target_size=(1600, 900)):
    # Create the output folder if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through every file in the input folder
    for filename in os.listdir(input_dir):
        # Only process image files
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Read the image
            img = cv2.imread(input_path)
            
            if img is None:
                print(f"⚠️ Could not read {filename}, skipping...")
                continue

            # Resize the image (OpenCV expects Width x Height)
            resized_img = cv2.resize(img, target_size)

            # Save to the new folder
            cv2.imwrite(output_path, resized_img)
            print(f"✅ Resized and saved: {filename}")

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # Change these paths to match where your images actually are
    INPUT_FOLDER = './cus4'
    OUTPUT_FOLDER = './custom_resized_1600x900'
    
    print(f"Starting resize process...")
    print(f"Input: {INPUT_FOLDER}")
    print(f"Output: {OUTPUT_FOLDER}\n")
    
    batch_resize(INPUT_FOLDER, OUTPUT_FOLDER)
    
    print("\n🎉 All images resized successfully!")