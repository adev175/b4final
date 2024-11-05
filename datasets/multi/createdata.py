start_index = 20
end_index = 643

# Generate the image paths
image_paths = [f'{str(i).zfill(5)}.png' for i in range(start_index, end_index + 1)]
folder_paths_GOPRO = ['F:\snufilm-test\\test\GOPRO_test\GOPR0384_11_00/', 'F:\snufilm-test\\test\GOPRO_test\GOPR0384_11_05/',
                'F:\snufilm-test\\test\GOPRO_test\GOPR0385_11_01/', 'F:\snufilm-test\\test\GOPRO_test\GOPR0396_11_00/',
                'F:\snufilm-test\\test\GOPRO_test\GOPR0410_11_00/', 'F:\snufilm-test\\test\GOPRO_test\GOPR0854_11_00/',
                'F:\snufilm-test\\test\GOPRO_test\GOPR0862_11_00/', 'F:\snufilm-test\\test\GOPRO_test\GOPR0868_11_00/',
                'F:\snufilm-test\\test\GOPRO_test\GOPR0869_11_00/', 'F:\snufilm-test\\test\GOPRO_test\GOPR0871_11_00/',
                'F:\snufilm-test\\test\GOPRO_test\GOPR0881_11_01/']
folder_paths_youtube = ["F:\snufilm-test\\test\YouTube_test\\0000/", "F:\snufilm-test\\test\YouTube_test\\0001",
                        "F:\snufilm-test\\test\YouTube_test\\0002", "F:\snufilm-test\\test\YouTube_test\\0003",
                        "F:\snufilm-test\\test\YouTube_test\\0004", "F:\snufilm-test\\test\YouTube_test\\0005",
                        "F:\snufilm-test\\test\YouTube_test\\0006", "F:\snufilm-test\\test\YouTube_test\\0007",
                        "F:\snufilm-test\\test\YouTube_test\\0008", "F:\snufilm-test\\test\YouTube_test\\0009",
                        "F:\snufilm-test\\test\YouTube_test\\0010", "F:\snufilm-test\\test\YouTube_test\\0011",
                        "F:\snufilm-test\\test\YouTube_test\\0012", "F:\snufilm-test\\test\YouTube_test\\0013",
                        "F:\snufilm-test\\test\YouTube_test\\0014", "F:\snufilm-test\\test\YouTube_test\\0015",
                        "F:\snufilm-test\\test\YouTube_test\\0016", "F:\snufilm-test\\test\YouTube_test\\0017",
                        "F:\snufilm-test\\test\YouTube_test\\0018", "F:\snufilm-test\\test\YouTube_test\\0019"]
# Print the image paths
og_list = 'F:\Pycharm Projects\MVFI_ANH\data\snufilm\multi_test2.txt'
folder_paths = folder_paths_youtube
with open(og_list, 'a') as f:
    for j in range(0,1):
        # Adjust the loop range to ensure it doesn't go out of bounds
        for i in range(0, len(image_paths), 4):
            # Check if the index is within bounds before accessing the list
            if i + 4 < len(image_paths):
                line = folder_paths[j] + image_paths[i] + ' ' + \
                       folder_paths[j] + image_paths[i + 1] + ' ' + \
                       folder_paths[j] + image_paths[i + 2] + ' ' + \
                       folder_paths[j] + image_paths[i + 3] + ' ' + \
                       folder_paths[j] + image_paths[i + 4] + '\n'
                f.write(line)
            else:
                # Handle the case where the index is out of bounds
                print("Index out of bounds. Skipping.")