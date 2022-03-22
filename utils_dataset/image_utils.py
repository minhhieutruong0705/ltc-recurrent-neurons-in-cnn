def crop_by_ratio(img, h_ratio, w_ratio):
    assert h_ratio <= 1 and w_ratio <= 1
    h, w = img.shape[:2]

    padding_h = int((1 - h_ratio) * h // 2)
    padding_w = int((1 - w_ratio) * w // 2)

    return img[padding_h:(h - padding_h), padding_w:(w - padding_w)]
