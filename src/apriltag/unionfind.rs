#[derive(Debug, Clone, Copy)]
pub struct Cluster {
    pub start_idx: usize,
    pub end_idx: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: u16,
    pub y: u16,
    pub gx: i16,
    pub gy: i16,
}

#[derive(Debug, Clone, Copy)]
pub struct Run {
    pub x_start: u16,
    pub x_end: u16,
    pub color: u8,
    pub id: u32,
}

/// A Run-Length based Union-Find implementation tailored for `AprilTag`.
pub struct UnionFind {
    /// Parent node for each run element.
    pub parent: Vec<u32>,
    /// The absolute pixel count (size) of the tree.
    pub size: Vec<u32>,
    /// Persistent arena for boundary points.
    pub edge_buffer: Vec<(u64, Point)>,
    /// Flat buffer of all extracted horizontal runs.
    pub runs: Vec<Run>,
    /// Indices indicating where each row starts in the `runs` vector.
    pub row_starts: Vec<usize>,
    /// A persistent buffer for the current row's run IDs, used to avoid heap allocation during gradient clustering.
    pub row_y_runs: Vec<u32>,
    /// A persistent buffer for the next row's run IDs, used in conjunction with `row_y_runs` for row-by-row scanning.
    pub row_y1_runs: Vec<u32>,
}

impl UnionFind {
    /// Creates a new `UnionFind` structure.
    pub fn new() -> Self {
        Self {
            parent: Vec::with_capacity(20_000),
            size: Vec::with_capacity(20_000),
            runs: Vec::with_capacity(20_000),
            row_starts: Vec::with_capacity(1000),
            edge_buffer: Vec::with_capacity(250_000),
            row_y_runs: vec![u32::MAX; 1296],
            row_y1_runs: vec![u32::MAX; 1296],
        }
    }

    /// Compresses all paths in the tree so lookups are instant
    pub fn flatten(&mut self) {
        let len = self.parent.len() as u32;
        for i in 0..len {
            self.get_representative(i);
        }
    }

    /// Resets the `UnionFind` structure so it can be reused without reallocating.
    pub fn clear(&mut self) {
        self.parent.clear();
        self.size.clear();
        self.runs.clear();
        self.row_starts.clear();
        self.edge_buffer.clear();
    }

    /// Finds the representative (root) of the set containing `id`.
    pub fn get_representative(&mut self, mut id: u32) -> u32 {
        let mut idx = id as usize;

        while self.parent[idx] != id {
            let parent_id = self.parent[idx];
            let grandparent_id = self.parent[parent_id as usize];

            self.parent[idx] = grandparent_id;
            id = grandparent_id;
            idx = id as usize;
        }

        id
    }

    /// Returns the number of elements in the set containing `id`.
    #[allow(dead_code)]
    pub fn get_set_size(&mut self, id: u32) -> u32 {
        let repid = self.get_representative(id);
        self.size[repid as usize]
    }

    /// Connects (unions) the sets containing `aid` and `bid`.
    /// Returns the representative of the newly merged set.
    pub fn connect(&mut self, aid: u32, bid: u32) -> u32 {
        let aroot = self.get_representative(aid);
        let broot = self.get_representative(bid);

        if aroot == broot {
            return aroot;
        }

        let asize = self.size[aroot as usize];
        let bsize = self.size[broot as usize];

        if asize > bsize {
            self.parent[broot as usize] = aroot;
            self.size[aroot as usize] += bsize;
            aroot
        } else {
            self.parent[aroot as usize] = broot;
            self.size[broot as usize] += asize;
            broot
        }
    }

    /// Extracts runs and connects them using strictly 4-way for black and 8-way for white.
    pub fn connected_components(&mut self, im: &[u8], w: usize, h: usize) {
        self.clear();
        const BG_CHUNK: u64 = 0x7F7F_7F7F_7F7F_7F7F;

        for y in 0..h {
            self.row_starts.push(self.runs.len());
            let y_offset = y * w;
            let mut x = 0;

            let prev_start = if y > 0 { self.row_starts[y - 1] } else { 0 };
            let prev_end = if y > 0 { self.row_starts[y] } else { 0 };
            let mut prev_idx = prev_start;

            while x < w {
                if x + 8 <= w {
                    let chunk =
                        u64::from_ne_bytes(im[y_offset + x..y_offset + x + 8].try_into().unwrap());
                    if chunk == BG_CHUNK {
                        x += 8;
                        continue;
                    }
                }

                let v = im[y_offset + x];
                if v == 127 {
                    x += 1;
                    continue;
                }

                let color = v;
                let start_x = x;
                x += 1;

                while x < w && im[y_offset + x] == color {
                    x += 1;
                }
                let end_x = x - 1;

                let run_id = self.parent.len() as u32;
                self.parent.push(run_id);
                self.size.push((end_x - start_x + 1) as u32);
                self.runs.push(Run {
                    x_start: start_x as u16,
                    x_end: end_x as u16,
                    color,
                    id: run_id,
                });

                if y > 0 {
                    let expand = usize::from(color == 255);
                    let match_start = start_x.saturating_sub(expand) as u16;
                    let match_end = (end_x + expand).min(w - 1) as u16;

                    while prev_idx < prev_end && self.runs[prev_idx].x_end < match_start {
                        prev_idx += 1;
                    }

                    let mut scan_idx = prev_idx;
                    while scan_idx < prev_end && self.runs[scan_idx].x_start <= match_end {
                        if self.runs[scan_idx].color == color {
                            self.connect(run_id, self.runs[scan_idx].id);
                        }
                        scan_idx += 1;
                    }
                }
            }
        }
        self.row_starts.push(self.runs.len());
    }

    #[inline(always)]
    fn fill_run_buffer(&self, buffer: &mut [u32], y: usize) {
        buffer.fill(u32::MAX);
        let start = self.row_starts[y];
        let end = self.row_starts[y + 1];
        for run in &self.runs[start..end] {
            for x in run.x_start..=run.x_end {
                buffer[x as usize] = run.id;
            }
        }
    }

    /// Extract boundary clusters (gradient boundary points)
    pub fn gradient_clusters(&mut self, im: &[u8], w: usize, h: usize) -> Vec<Cluster> {
        self.edge_buffer.clear();
        self.flatten();

        let mut row_y_runs = std::mem::take(&mut self.row_y_runs);
        let mut row_y1_runs = std::mem::take(&mut self.row_y1_runs);

        if row_y_runs.len() < w {
            row_y_runs.resize(w, u32::MAX);
        }
        if row_y1_runs.len() < w {
            row_y1_runs.resize(w, u32::MAX);
        }

        self.fill_run_buffer(&mut row_y_runs, 0);

        const BG_CHUNK: u64 = 0x7F7F_7F7F_7F7F_7F7F;

        for y in 0..(h - 1) {
            self.fill_run_buffer(&mut row_y1_runs, y + 1);
            let mut connected_last = false;

            let row_im = &im[y * w..(y + 1) * w];
            let next_row_im = &im[(y + 1) * w..(y + 2) * w];

            let mut x = 1;
            while x < w - 1 {
                if x + 8 < w {
                    let chunk = u64::from_ne_bytes(row_im[x..x + 8].try_into().unwrap());
                    if chunk == BG_CHUNK {
                        x += 8;
                        connected_last = false;
                        continue;
                    }
                }

                let v0 = row_im[x];
                if v0 == 127 {
                    connected_last = false;
                    x += 1;
                    continue;
                }

                let run0 = row_y_runs[x];
                if run0 == u32::MAX {
                    connected_last = false;
                    x += 1;
                    continue;
                }

                let rep0 = self.parent[run0 as usize];
                let size0 = self.size[rep0 as usize];

                if size0 < 25 {
                    connected_last = false;
                    x += 1;
                    continue;
                }

                let mut connected = false;

                macro_rules! check_neighbor {
                    ($dx:expr, $dy:expr, $v1:expr, $run_arr:expr, $nx:expr) => {
                        if (v0 ^ $v1) == 255 {
                            let run1 = $run_arr[$nx];
                            if run1 != u32::MAX {
                                let rep1 = self.parent[run1 as usize];
                                if self.size[rep1 as usize] >= 25 {
                                    let clusterid = if rep0 < rep1 {
                                        u64::from(rep1) << 32 | u64::from(rep0)
                                    } else {
                                        u64::from(rep0) << 32 | u64::from(rep1)
                                    };

                                    let gx = $dx as i16 * (i16::from($v1) - i16::from(v0));
                                    let gy = $dy as i16 * (i16::from($v1) - i16::from(v0));

                                    self.edge_buffer.push((
                                        clusterid,
                                        Point {
                                            x: (2 * x as isize + $dx) as u16,
                                            y: (2 * y as isize + $dy) as u16,
                                            gx,
                                            gy,
                                        },
                                    ));
                                    connected = true;
                                }
                            }
                        }
                    };
                }

                // Right
                let v_right = row_im[x + 1];
                check_neighbor!(1, 0, v_right, row_y_runs, x + 1);

                // Down
                let v_down = next_row_im[x];
                check_neighbor!(0, 1, v_down, row_y1_runs, x);

                // Down-Left
                if !connected_last {
                    let v_down_left = next_row_im[x - 1];
                    check_neighbor!(-1, 1, v_down_left, row_y1_runs, x - 1);
                }

                // Down-Right
                let v_down_right = next_row_im[x + 1];
                check_neighbor!(1, 1, v_down_right, row_y1_runs, x + 1);

                connected_last = connected;
                x += 1;
            }

            std::mem::swap(&mut row_y_runs, &mut row_y1_runs);
        }

        self.row_y_runs = row_y_runs;
        self.row_y1_runs = row_y1_runs;

        radsort::sort_by_key(&mut self.edge_buffer, |&(id, _)| id);

        let mut clusters = Vec::new();
        let mut chunk_start = 0;

        while chunk_start < self.edge_buffer.len() {
            let current_id = self.edge_buffer[chunk_start].0;
            let mut chunk_end = chunk_start + 1;

            while chunk_end < self.edge_buffer.len() && self.edge_buffer[chunk_end].0 == current_id
            {
                chunk_end += 1;
            }

            if chunk_end - chunk_start >= 24 {
                clusters.push(Cluster {
                    start_idx: chunk_start,
                    end_idx: chunk_end,
                });
            }

            chunk_start = chunk_end;
        }

        clusters
    }
}
