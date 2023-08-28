import os
def create_subfolder(path, folder_name):
    """
    Create a subfolder with the given folder_name inside the directory specified by path.

    Parameters:
    path (str): The path of the parent folder where the new folder should be created.
    folder_name (str): The name of the new folder to be created.

    Returns:
    str: The path of the newly created folder.
    """
    try:
        # Combine the parent path and the folder name to get the full path
        full_path = os.path.join(path, folder_name)

        # Create the folder
        os.makedirs(full_path, exist_ok=True)

        print(f"Folder '{folder_name}' created at {full_path}")
        return full_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
# # Test
# path = r"C:\Users\xmoot\Desktop\Models\TensorFlow\ssl-seizure-detection\models"
# folder_name = "test"

# create_subfolder(path, folder_name)
