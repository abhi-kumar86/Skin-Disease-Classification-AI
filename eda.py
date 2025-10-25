import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# CONFIGURATION part
DATA_DIR = 'dataset'
CLASSES = ['acne', 'hyperpigmentation', 'nail_psoriasis', 'sjs_ten', 'vitiligo']
OUTPUT_DIR = 'outputs'

# Class descriptions
CLASS_INFO = {
    'acne': 'Acne - Common skin condition with pimples',
    'hyperpigmentation': 'Hyperpigmentation - Dark patches on skin',
    'nail_psoriasis': 'Nail Psoriasis - Nail disorder',
    'sjs_ten': 'SJS-TEN - Stevens-Johnson Syndrome',
    'vitiligo': 'Vitiligo - Loss of skin pigmentation'
}

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("EXPLORATORY DATA ANALYSIS - SKIN DISEASE CLASSIFICATION")
print("="*80)
print(f"Dataset Source: https://data.mendeley.com/datasets/3hckgznc67/1")
print(f"Number of Classes: {len(CLASSES)}")
print("="*80)

# COUNTING IMAGES
print("\n[STEP 1] Counting Images in Each Class...")

class_counts = {}
all_image_paths = {}

for cls in CLASSES:
    cls_path = os.path.join(DATA_DIR, cls)
    
    if not os.path.exists(cls_path):
        print(f"⚠️  WARNING: Folder '{cls}' not found at {cls_path}")
        class_counts[cls] = 0
        all_image_paths[cls] = []
        continue
    
    # Get all image files
    images = [f for f in os.listdir(cls_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    class_counts[cls] = len(images)
    all_image_paths[cls] = [os.path.join(cls_path, img) for img in images[:100]]  # Store first 100
    
    print(f"✅ {cls:20s}: {len(images):5d} images")

total_images = sum(class_counts.values())
print(f"\n📊 Total Images: {total_images}")

if total_images == 0:
    print("\n❌ ERROR: No images found! Please check your dataset folder structure.")
    print("Expected structure:")
    print("dataset/")
    print("├── acne/")
    print("├── hyperpigmentation/")
    print("├── nail_psoriasis/")
    print("├── sjs_ten/")
    print("└── vitiligo/")
    exit(1)

# CLASS DISTRIBUTION part
print("\n" + "="*80)
print("[STEP 2] Class Distribution Analysis")
print("="*80)

# Calculate percentages
for cls, count in class_counts.items():
    if total_images > 0:
        percentage = (count / total_images) * 100
        print(f"{cls:20s}: {count:5d} images ({percentage:5.2f}%)")

# Check for class imbalance
counts_list = list(class_counts.values())
if max(counts_list) > 0 and min(counts_list) > 0:
    imbalance_ratio = max(counts_list) / min(counts_list)
    print(f"\n⚖️  Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print("⚠️  Moderate class imbalance detected!")
        print("   Recommendation: Use class weights during training")

# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
colors = plt.cm.Set3(range(len(CLASSES)))
axes[0].bar(range(len(CLASSES)), [class_counts[c] for c in CLASSES], 
            color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_xticks(range(len(CLASSES)))
axes[0].set_xticklabels(CLASSES, rotation=45, ha='right')
axes[0].set_xlabel('Disease Class', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
axes[0].set_title('Class Distribution (Bar Chart)', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for i, cls in enumerate(CLASSES):
    count = class_counts[cls]
    axes[0].text(i, count + max(counts_list)*0.02, str(count), 
                ha='center', va='bottom', fontweight='bold', fontsize=10)

# Pie chart
axes[1].pie([class_counts[c] for c in CLASSES],
            labels=[f"{c}\n({class_counts[c]} images)" for c in CLASSES],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 9, 'weight': 'bold'})
axes[1].set_title('Class Proportions', fontsize=14, fontweight='bold')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'class_distribution.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Class distribution plot saved: {output_path}")
plt.close()

# SAMPLE IMAGES
print("\n" + "="*80)
print("[STEP 3] Generating Sample Image Grid")
print("="*80)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

sample_idx = 0
for cls in CLASSES:
    if class_counts[cls] == 0:
        continue
    
    # Get 2 random samples
    images_in_class = all_image_paths[cls]
    
    if len(images_in_class) > 0:
        # First sample
        img_path = images_in_class[0]
        try:
            img = Image.open(img_path).convert('RGB')
            axes[sample_idx].imshow(img)
            axes[sample_idx].set_title(f"{cls}\n({CLASS_INFO[cls][:30]}...)", 
                                      fontsize=9, fontweight='bold')
            axes[sample_idx].axis('off')
        except:
            axes[sample_idx].text(0.5, 0.5, 'Error loading', ha='center')
            axes[sample_idx].axis('off')
        
        sample_idx += 1
    
    if len(images_in_class) > 1:
        # Second sample
        img_path = images_in_class[min(1, len(images_in_class)-1)]
        try:
            img = Image.open(img_path).convert('RGB')
            axes[sample_idx].imshow(img)
            axes[sample_idx].set_title(f"{cls} (Sample 2)", 
                                      fontsize=9, fontweight='bold')
            axes[sample_idx].axis('off')
        except:
            axes[sample_idx].text(0.5, 0.5, 'Error loading', ha='center')
            axes[sample_idx].axis('off')
        
        sample_idx += 1

# Hide extra subplots
for i in range(sample_idx, 10):
    axes[i].axis('off')

plt.suptitle('Sample Images from Each Disease Class', fontsize=16, fontweight='bold')
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'sample_images.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Sample images saved: {output_path}")
plt.close()

# ==================== SECTION 4: IMAGE PROPERTIES ====================
print("\n" + "="*80)
print("[STEP 4] Analyzing Image Properties")
print("="*80)

print("Analyzing sample images...")
image_widths = []
image_heights = []
image_aspects = []

sample_size = min(50, total_images)  # Analyze 50 images
analyzed = 0

for cls in CLASSES:
    if class_counts[cls] == 0:
        continue
    
    for img_path in all_image_paths[cls][:20]:  # 20 per class max
        try:
            img = Image.open(img_path)
            width, height = img.size
            image_widths.append(width)
            image_heights.append(height)
            image_aspects.append(width / height)
            analyzed += 1
            
            if analyzed >= sample_size:
                break
        except:
            continue
    
    if analyzed >= sample_size:
        break

if image_widths:
    print(f"\n📐 Image Statistics (from {analyzed} samples):")
    print(f"   Width  - Mean: {np.mean(image_widths):.0f}px, Range: {min(image_widths)}-{max(image_widths)}px")
    print(f"   Height - Mean: {np.mean(image_heights):.0f}px, Range: {min(image_heights)}-{max(image_heights)}px")
    print(f"   Aspect Ratio - Mean: {np.mean(image_aspects):.2f}")
    
    # Plot image size distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(image_widths, bins=20, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Width (pixels)', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Image Width Distribution', fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(image_heights, bins=20, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Height (pixels)', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Image Height Distribution', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    axes[2].hist(image_aspects, bins=20, color='lightgreen', edgecolor='black')
    axes[2].set_xlabel('Aspect Ratio (W/H)', fontweight='bold')
    axes[2].set_ylabel('Frequency', fontweight='bold')
    axes[2].set_title('Aspect Ratio Distribution', fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'image_properties.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Image properties plot saved: {output_path}")
    plt.close()

# FINAL EDA  SUMMARY REPORT
print("\n" + "="*80)
print("EDA SUMMARY REPORT")
print("="*80)

print(f"\n📊 Dataset Overview:")
print(f"   Total Images: {total_images}")
print(f"   Number of Classes: {len(CLASSES)}")
print(f"   Classes: {', '.join(CLASSES)}")

print(f"\n🏥 Class Statistics:")
if counts_list and max(counts_list) > 0:
    max_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=class_counts.get)
    print(f"   Largest Class: {max_class} ({class_counts[max_class]} images)")
    print(f"   Smallest Class: {min_class} ({class_counts[min_class]} images)")
    if min(counts_list) > 0:
        print(f"   Imbalance Ratio: {max(counts_list)/min(counts_list):.2f}:1")

print(f"\n✅ Generated Files in '{OUTPUT_DIR}/':")
for file in sorted(os.listdir(OUTPUT_DIR)):
    print(f"   - {file}")

print("\n" + "="*80)
print("✅ EDA COMPLETED SUCCESSFULLY!")
print("="*80)
print("\n📋 Next Steps:")
print("1. Review the generated plots in 'outputs/' folder")
print("2. Check class balance - consider data augmentation if needed")
print("3. Run training: python train.py")

print("="*80)
