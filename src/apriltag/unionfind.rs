/// Represents a horizontal run of identical pixels.
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub y: u16,
    pub x_start: u16,
    pub x_end: u16,
    pub parent: u32,
    pub size: u32,
}

pub struct Blob {
    pub root_id: u32,
    pub pixel_count: u32,
    pub start_idx: usize,
    pub end_idx: usize,
}

pub struct RleUnionFind {
    pub segments: Vec<Segment>,
    prev_row_start: usize,
    prev_row_end: usize,
    curr_row_start: usize,
    current_y: u16,
}

impl RleUnionFind {
    pub fn new(capacity: usize) -> Self {
        Self {
            segments: Vec::with_capacity(capacity),
            prev_row_start: 0,
            prev_row_end: 0,
            curr_row_start: 0,
            current_y: 0,
        }
    }

    /// Resets the global state before processing a new camera frame.
    pub fn clear(&mut self) {
        self.segments.clear();
        self.prev_row_start = 0;
        self.prev_row_end = 0;
        self.curr_row_start = 0;
        self.current_y = 0;
    }

    /// Processes a single DMA chunk of a thresholded monochrome image.
    /// `chunk`: Flat slice of the binary image pixels.
    /// `width`: Image width (e.g., 640).
    /// `height`: Number of rows in this specific chunk.
    /// `y_offset`: The global y-coordinate of the first row in this chunk.
    pub fn process_chunk(&mut self, chunk: &[u8], width: usize, height: usize, y_offset: usize) {
        let start_idx = self.segments.len();

        // Step 1: Extract segments for this chunk
        for local_y in 0..height {
            let y = (y_offset + local_y) as u16;
            let row_start = local_y * width;
            let row = &chunk[row_start..row_start + width];

            let mut in_segment = false;
            let mut x_start = 0;

            for x in 0..width {
                let is_active = row[x] == 255;

                if is_active && !in_segment {
                    x_start = x as u16;
                    in_segment = true;
                } else if !is_active && in_segment {
                    self.push_segment(x_start, (x - 1) as u16, y);
                    in_segment = false;
                }
            }

            if in_segment {
                self.push_segment(x_start, (width - 1) as u16, y);
            }
        }

        // Step 2: Connect the newly added segments, carrying over state from previous chunks
        for i in start_idx..self.segments.len() {
            let curr_seg = self.segments[i];

            if curr_seg.y > self.current_y {
                self.prev_row_start = self.curr_row_start;
                self.prev_row_end = i;
                self.curr_row_start = i;
                self.current_y = curr_seg.y;
            }

            if curr_seg.y == 0 {
                continue;
            }

            if self.prev_row_start < self.prev_row_end
                && self.segments[self.prev_row_start].y == curr_seg.y - 1
            {
                while self.prev_row_start < self.prev_row_end
                    && self.segments[self.prev_row_start].x_end + 1 < curr_seg.x_start
                {
                    self.prev_row_start += 1;
                }

                let mut j = self.prev_row_start;
                while j < self.prev_row_end {
                    let prev_seg = self.segments[j];

                    if prev_seg.x_start > curr_seg.x_end + 1 {
                        break;
                    }

                    self.union(i as u32, j as u32);
                    j += 1;
                }
            }
        }
    }

    #[inline(always)]
    fn push_segment(&mut self, x_start: u16, x_end: u16, y: u16) {
        let idx = self.segments.len() as u32;
        self.segments.push(Segment {
            y,
            x_start,
            x_end,
            parent: idx,
            size: (x_end - x_start + 1) as u32,
        });
    }

    /// Finds the root of a segment's set with Path Compression.
    pub fn find(&mut self, i: u32) -> u32 {
        let mut root = i;
        while root != self.segments[root as usize].parent {
            root = self.segments[root as usize].parent;
        }

        let mut curr = i;
        while curr != root {
            let nxt = self.segments[curr as usize].parent;
            self.segments[curr as usize].parent = root;
            curr = nxt;
        }
        root
    }

    /// Unions two disjoint sets by rank/size.
    pub fn union(&mut self, i: u32, j: u32) {
        let root_i = self.find(i);
        let root_j = self.find(j);

        if root_i != root_j {
            let size_i = self.segments[root_i as usize].size;
            let size_j = self.segments[root_j as usize].size;

            if size_i < size_j {
                self.segments[root_i as usize].parent = root_j;
                self.segments[root_j as usize].size += size_i;
            } else {
                self.segments[root_j as usize].parent = root_i;
                self.segments[root_i as usize].size += size_j;
            }
        }
    }

    /// Step 3: Flatten the tree so every segment points directly to its blob root.
    /// Call this ONCE after all chunks have been processed.
    pub fn flatten(&mut self) -> usize {
        let mut unique_blobs = 0;
        for i in 0..self.segments.len() {
            let root = self.find(i as u32);
            self.segments[i].parent = root;
            if root == i as u32 {
                unique_blobs += 1;
            }
        }
        unique_blobs
    }

    /// Step 4: Group segments into blobs and filter out invalid sizes.
    /// Returns a list of valid blobs ready for contour tracing.
    pub fn extract_valid_blobs(&mut self, min_pixels: u32, max_pixels: u32) -> Vec<Blob> {
        self.segments.sort_unstable_by_key(|s| s.parent);

        let mut valid_blobs = Vec::new();
        if self.segments.is_empty() {
            return valid_blobs;
        }

        let mut current_root = self.segments[0].parent;
        let mut start_idx = 0;
        let mut pixel_count = 0;

        for i in 0..self.segments.len() {
            let seg = self.segments[i];

            if seg.parent != current_root {
                if pixel_count >= min_pixels && pixel_count <= max_pixels {
                    valid_blobs.push(Blob {
                        root_id: current_root,
                        pixel_count,
                        start_idx,
                        end_idx: i,
                    });
                }

                current_root = seg.parent;
                start_idx = i;
                pixel_count = 0;
            }

            pixel_count += (seg.x_end - seg.x_start + 1) as u32;
        }

        if pixel_count >= min_pixels && pixel_count <= max_pixels {
            valid_blobs.push(Blob {
                root_id: current_root,
                pixel_count,
                start_idx,
                end_idx: self.segments.len(),
            });
        }

        valid_blobs
    }
}
