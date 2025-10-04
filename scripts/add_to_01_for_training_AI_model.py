# (Keep all your existing imports and helper functions as they are)

# train_model.py

# 1. READ THE DATA
# The script will read your 'selections_log.csv' file.
# It will load the images and create pairs for training:
# (your_chosen_image, one_of_the_other_two)
# It will label these pairs to teach the model that your chosen image is "better."

# 2. LOAD A PRE-TRAINED MODEL
# It will use a pre-trained model like a Vision Transformer (ViT).
# This is called transfer learning. It's like giving a student an advanced textbook 
# instead of having them learn to read from scratch.

# 3. FINE-TUNE THE MODEL
# It will train the model using a "triplet loss" function. This loss function
# is designed to bring the embeddings of your chosen images closer together 
# in a feature space while pushing the embeddings of the "not chosen" images away.
# 

# 4. SAVE THE TRAINED MODEL
# Once training is complete, the script will save the final model.
# This saved model can then be used in your main workflow to automatically 
# rank new sets of images.


def main():
    parser = argparse.ArgumentParser(description="Review 3 images; keep 1 and log the selection.")
    parser.add_argument("folder", type=str, help="Folder containing images")
    parser.add_argument("--exts", type=str, default="png", help="Comma-separated list of extensions to include")
    parser.add_argument("--print-triplets", action="store_true", help="Print grouped triplets and exit (debugging aid)")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        human_err(f"Folder not found: {folder}")
        sys.exit(1)

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    files = scan_images(folder, exts)
    if not files:
        human_err("No images found. Check --exts or folder path.")
        sys.exit(1)

    triplets = find_triplets(files)
    if not triplets:
        human_err("No triplets found with the current grouping. Try a different flag or no grouping.")
        sys.exit(1)

    if args.print_triplets:
        for idx, t in enumerate(triplets, 1):
            print(f"\nTriplet {idx}:")
            for p in t:
                print("  -", p.name)
        print(f"\nTotal triplets: {len(triplets)}")
        return

    # === NEW: Set up a dedicated training data directory ===
    training_data_dir = folder.parent / "training_data"
    training_data_dir.mkdir(exist_ok=True)
    log_path = training_data_dir / "selections_log.csv"
    
    # Initialize log file with header if it doesn't exist
    header_needed = not log_path.exists()
    if header_needed:
        with log_path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["set_id", "chosen_path", "image_1_path", "image_2_path", "image_3_path"])

    index = 0
    total_triplets = len(triplets)
    print(f"[*] Found {total_triplets} triplets. Starting at {folder}")

    while 0 <= index < total_triplets:
        t = triplets[index]
        remaining = total_triplets - index
        
        memory_level = check_memory_warning()
        memory_display = format_memory_display()
        
        if memory_level > 0:
            print(f"\n⚠️  MEMORY WARNING{memory_display}")
            
        print(f"\n=== Triplet {index+1}/{total_triplets} • {remaining} remaining{memory_display} ===")
        for i, p in enumerate(t, start=1):
            print(f"{i}. {p.name}")

        if _MEMORY_MONITORING_AVAILABLE and memory_level > 0:
            gc.collect()

        choice = show_triplet(t, current=index + 1, total=total_triplets)

        if choice in (0, 1, 2):
            # === NEW: Don't delete or move. Just log the selection. ===
            chosen_path = t[choice]
            set_id = chosen_path.stem.split('_')[0]
            
            # This is the crucial part: log the full paths of all three images
            with log_path.open('a', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    set_id,
                    str(chosen_path),
                    str(t[0]),
                    str(t[1]),
                    str(t[2])
                ])
                
            info(f"Logged selection for {set_id}. Chosen: {chosen_path.name}")
            
        elif choice == -4:  # Delete all three images
            try:
                # Retain the old behavior for "delete all" as it implies "terrible"
                safe_delete(list(t), hard_delete=args.hard_delete)
                # You might want to log this as well, as it's a valuable signal
                with log_path.open('a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([f"set_{index+1}", "DELETED_ALL", str(t[0]), str(t[1]), str(t[2])])
                
            except RuntimeError as e:
                human_err(str(e))
                print("Aborting due to deletion method issue.")
                break

        elif choice == -3:  # quit
            print("Quitting.")
            break
        
        index += 1

    print("\nDone. All selections logged to:", log_path)


if __name__ == "__main__":
    main()