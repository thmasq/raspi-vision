use std::collections::HashMap;

pub struct Cluster {
    pub id: u64,
    pub points: Vec<Point>,
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: u16,
    pub y: u16,
    pub gx: i16,
    pub gy: i16,
}

/// A pixel-based Union-Find implementation.
pub struct UnionFind {
    pub maxid: u32,
    /// Parent node for each element. Initialized to u32::MAX.
    pub parent: Vec<u32>,
    /// The size of the tree excluding the root.
    pub size: Vec<u32>,
}

impl UnionFind {
    /// Creates a new UnionFind structure capable of holding up to `maxid` elements.
    pub fn new(maxid: u32) -> Self {
        let len = (maxid + 1) as usize;
        Self {
            maxid,
            parent: vec![u32::MAX; len],
            size: vec![0; len],
        }
    }

    /// Resets the UnionFind structure so it can be reused without reallocating.
    pub fn clear(&mut self) {
        self.parent.fill(u32::MAX);
        self.size.fill(0);
    }

    /// Finds the representative (root) of the set containing `id`.
    /// Includes "Path Halving" optimization and lazy initialization.
    pub fn get_representative(&mut self, mut id: u32) -> u32 {
        let mut idx = id as usize;

        if self.parent[idx] == u32::MAX {
            self.parent[idx] = id;
            return id;
        }

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
    pub fn get_set_size(&mut self, id: u32) -> u32 {
        let repid = self.get_representative(id);
        self.size[repid as usize] + 1
    }

    /// Connects (unions) the sets containing `aid` and `bid`.
    /// Returns the representative of the newly merged set.
    pub fn connect(&mut self, aid: u32, bid: u32) -> u32 {
        let aroot = self.get_representative(aid);
        let broot = self.get_representative(bid);

        if aroot == broot {
            return aroot;
        }

        let asize = self.size[aroot as usize] + 1;
        let bsize = self.size[broot as usize] + 1;

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

    /// 4-connected union find on the thresholded image pixels
    pub fn connected_components(&mut self, im: &[u8], w: usize, h: usize) {
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let v = im[idx];

                if v == 127 {
                    continue;
                }

                if x > 0 && im[idx - 1] == v {
                    self.connect(idx as u32, (idx - 1) as u32);
                }
                if y > 0 && im[idx - w] == v {
                    self.connect(idx as u32, (idx - w) as u32);
                }

                if v == 255 {
                    if x > 0 && y > 0 && im[idx - w - 1] == v {
                        self.connect(idx as u32, (idx - w - 1) as u32);
                    }
                    if x + 1 < w && y > 0 && im[idx - w + 1] == v {
                        self.connect(idx as u32, (idx - w + 1) as u32);
                    }
                }
            }
        }
    }

    /// Extract boundary clusters (gradient boundary points)
    pub fn gradient_clusters(&mut self, im: &[u8], w: usize, h: usize) -> Vec<Cluster> {
        let mut map: HashMap<u64, Vec<Point>> = HashMap::new();

        for y in 0..(h - 1) {
            let mut connected_last = false;

            for x in 1..(w - 1) {
                let idx0 = y * w + x;
                let v0 = im[idx0];

                if v0 == 127 {
                    connected_last = false;
                    continue;
                }

                let rep0 = self.get_representative(idx0 as u32);
                if self.get_set_size(rep0) < 25 {
                    connected_last = false;
                    continue;
                }

                let mut check_conn = |dx: isize, dy: isize| -> bool {
                    let nx = (x as isize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    let idx1 = ny * w + nx;
                    let v1 = im[idx1];

                    if v1 != 127 && (v0 as u16 + v1 as u16) == 255 {
                        let rep1 = self.get_representative(idx1 as u32);
                        if self.get_set_size(rep1) >= 25 {
                            let clusterid = if rep0 < rep1 {
                                (rep1 as u64) << 32 | (rep0 as u64)
                            } else {
                                (rep0 as u64) << 32 | (rep1 as u64)
                            };

                            let gx = dx as i16 * (v1 as i16 - v0 as i16);
                            let gy = dy as i16 * (v1 as i16 - v0 as i16);

                            map.entry(clusterid).or_default().push(Point {
                                x: (2 * x as isize + dx) as u16,
                                y: (2 * y as isize + dy) as u16,
                                gx,
                                gy,
                            });

                            return true;
                        }
                    }
                    false
                };

                let mut connected = false;

                connected |= check_conn(1, 0);
                connected |= check_conn(0, 1);

                if !connected_last {
                    connected |= check_conn(-1, 1);
                }

                connected |= check_conn(1, 1);

                connected_last = connected;
            }
        }

        map.into_iter()
            .map(|(id, points)| Cluster { id, points })
            .collect()
    }
}
