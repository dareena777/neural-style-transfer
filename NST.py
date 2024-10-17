import tkinter as tk
import customtkinter as ctk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :])  # Keep only the first 3 channels (RGB)
])

class VGG(nn.Module):
    def __init__(self): 
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights='DEFAULT').features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

def neural_style_transfer(original, style, total_steps=6000, learning_rate=0.001, alpha=1, beta=0.01):
    generated = original.clone().requires_grad_(True)
    optimizer = optim.Adam([generated], lr=learning_rate)
    model = VGG().to(device).eval()

    for step in range(total_steps):
        generated_features = model(generated)
        original_features = model(original)
        style_features = model(style)

        style_loss = original_loss = 0
        for generated_feature, original_feature, style_feature in zip(generated_features, original_features, style_features):
            batch_size, channel, height, width = generated_feature.shape

            # Calculate original loss
            original_loss += torch.mean((generated_feature - original_feature) ** 2)

            # Gram matrix for generated feature
            g = generated_feature.view(channel, height * width).mm(generated_feature.view(channel, height * width).t())
            # Gram matrix for style feature
            a = style_feature.view(channel, height * width).mm(style_feature.view(channel, height * width).t())

            # Calculate style loss
            style_loss += torch.mean((g - a) ** 2)

        # Total loss
        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step: {step}, Total Loss: {total_loss.item()}")
            save_image(generated, f"generated_{step}.png")

    return generated

# GUI setup
ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('green')
root = ctk.CTk()
root.geometry('1080x720')
root.title('Neural Style Transfer')

image_data = None
image_data_2 = None

def open_image():
    global image_data, photo
    filename = askopenfilename()
    if filename:
        image_data = load_image(filename)
        display_image(image_data, "Original Image", 50)

def open_image_style():
    global image_data_2, photo2
    filename = askopenfilename()
    if filename:
        image_data_2 = load_image(filename)
        display_image(image_data_2, "Style Image", 350)

def display_image(image_tensor, label_text, x_position):
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    image_np = (image_np * 255).astype('uint8')
    image_pil = Image.fromarray(image_np)

    photo = ImageTk.PhotoImage(image_pil.resize((300, 300)))
    image_frame = ctk.CTkFrame(master=root, width=300, height=300)
    image_frame.pack(side=tk.LEFT, padx=10)
    image_frame.place(x=x_position, y=80)

    image_label = tk.Label(image_frame, image=photo)
    image_label.image = photo  # Keep a reference to avoid garbage collection
    image_label.pack()

    label = ctk.CTkLabel(master=root, text=label_text, font=("Roboto", 24))
    label.pack()
    label.place(x=x_position, y=30)

def run_style_transfer():
    if image_data is not None and image_data_2 is not None:
        generated_image = neural_style_transfer(image_data, image_data_2)
        save_image(generated_image, "final_output.png")
        print("Style transfer complete. Output saved as 'final_output.png'.")

open_button = ctk.CTkButton(master=root, text='Upload Image', command=open_image)
open_button.place(x=50, y=350)

open_button_style = ctk.CTkButton(master=root, text='Upload Style', command=open_image_style)
open_button_style.place(x=50, y=400)

run_button = ctk.CTkButton(master=root, text='Run Style Transfer', command=run_style_transfer)
run_button.place(x=50, y=450)

root.mainloop()
