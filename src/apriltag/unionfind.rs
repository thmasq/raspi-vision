use rayon::prelude::*;
use rustc_hash::FxHashMap;

#[allow(dead_code)]
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
    /// Parent node for each element. Initialized to `u32::MAX`.
    pub parent: Vec<u32>,
    /// The size of the tree excluding the root.
    pub size: Vec<u32>,
}

impl UnionFind {
    /// Creates a new `UnionFind` structure capable of holding up to `maxid` elements.
    pub fn new(maxid: u32) -> Self {
        let len = (maxid + 1) as usize;
        Self {
            maxid,
            parent: vec![u32::MAX; len],
            size: vec![0; len],
        }
    }

    /// Compresses all paths in the tree so lookups are instant
    pub fn flatten(&mut self) {
        for i in 0..self.maxid {
            if self.parent[i as usize] != u32::MAX {
                self.get_representative(i);
            }
        }
    }

    /// Read-only representative fetcher (Assumes tree is flattened)
    pub fn get_representative_readonly(&self, mut id: u32) -> u32 {
        let mut idx = id as usize;
        if self.parent[idx] == u32::MAX {
            return id;
        }
        while self.parent[idx] != id {
            id = self.parent[idx];
            idx = id as usize;
        }
        id
    }

    /// Resets the `UnionFind` structure so it can be reused without reallocating.
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
    #[allow(dead_code)]
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

    /// 4-connected (and partially 8-connected) union find on the thresholded image pixels
    pub fn connected_components(&mut self, im: &[u8], w: usize, h: usize) {
        const BG_CHUNK: u64 = 0x7F7F_7F7F_7F7F_7F7F;

        for y in 0..h {
            let mut x = 0;

            while x < w {
                let idx = y * w + x;

                if x + 8 <= w {
                    let chunk = u64::from_ne_bytes(im[idx..idx + 8].try_into().unwrap());
                    if chunk == BG_CHUNK {
                        x += 8;
                        continue;
                    }
                }

                let v = im[idx];

                if v == 127 {
                    x += 1;
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

                x += 1;
            }
        }
    }

    /// Extract boundary clusters (gradient boundary points)
    pub fn gradient_clusters(&self, im: &[u8], w: usize, h: usize) -> Vec<Cluster> {
        let y_indices: Vec<usize> = (0..(h - 1)).collect();

        let map = y_indices
            .into_par_iter()
            .fold(FxHashMap::<u64, Vec<Point>>::default, |mut local_map, y| {
                let mut connected_last = false;

                for x in 1..(w - 1) {
                    let idx0 = y * w + x;
                    let v0 = im[idx0];

                    if v0 == 127 {
                        connected_last = false;
                        continue;
                    }

                    let rep0 = self.get_representative_readonly(idx0 as u32);
                    let size0 = self.size[rep0 as usize] + 1;
                    if size0 < 25 {
                        connected_last = false;
                        continue;
                    }

                    let mut check_conn = |dx: isize, dy: isize| -> bool {
                        let nx = (x as isize + dx) as usize;
                        let ny = (y as isize + dy) as usize;
                        let idx1 = ny * w + nx;
                        let v1 = im[idx1];

                        if v1 != 127 && (u16::from(v0) + u16::from(v1)) == 255 {
                            let rep1 = self.get_representative_readonly(idx1 as u32);
                            let size1 = self.size[rep1 as usize] + 1;

                            if size1 >= 25 {
                                let clusterid = if rep0 < rep1 {
                                    u64::from(rep1) << 32 | u64::from(rep0)
                                } else {
                                    u64::from(rep0) << 32 | u64::from(rep1)
                                };

                                let gx = dx as i16 * (i16::from(v1) - i16::from(v0));
                                let gy = dy as i16 * (i16::from(v1) - i16::from(v0));

                                local_map.entry(clusterid).or_default().push(Point {
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
                local_map
            })
            .reduce(FxHashMap::default, |mut map1, map2| {
                for (k, mut v) in map2 {
                    map1.entry(k).or_default().append(&mut v);
                }
                map1
            });

        map.into_iter()
            .map(|(id, points)| Cluster { id, points })
            .collect()
    }
}
