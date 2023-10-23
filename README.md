# Feature

## SIFT: Scale Invariant Feature Transform

# Example

```
// load test image
const auuto path = "...";
const auto image = cv::imread(path, 0);

// detect SIFT features
dgelom::Sift sift(image.rows, image.cols);
auto feats = sift.detect(image.ptr(), true);

// Visualization
auto fmap = cv::imread(path);
for(const auto& f: feats){
   cv::circle(fmap, {int(f.x), int(f.y)}, f.scale*5, {255, 255, 0}, 2);
}
```
