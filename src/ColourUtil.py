class ColourCheck:
    TARGET_COLORS = {"Red": (255, 0, 0), "Yellow": (255, 255, 0), "Green": (0, 255, 0)}

    def color_difference(self, colour2):
        return sum([abs(component1-component2) for component1, component2 in zip(self, colour2)])

    def get_pixels(self, x, y):
        red_image_rgb = self.convert("RGB")

        rgb_pixel_value = red_image_rgb.getpixel((x, y))
        return rgb_pixel_value

    # my_color = (201, 206, 212)
    #
    # differences = [[color_difference(my_color, target_value), target_name]
    #                for target_name, target_value in TARGET_COLORS.items()]
    # differences.sort()  # sorted by the first element of inner lists
    # my_color_name = differences[0][1]
    #
    # print(my_color_name)