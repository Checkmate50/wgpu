use obj::{load_obj, Obj};
use std::fs::File;
use std::io::BufReader;

pub fn load_model(file_name: &str) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u16>) {
    let input = BufReader::new(
        File::open(file_name).expect(&format!("{} is not a file I could find", file_name)),
    );
    let dome: Obj =
        load_obj(input).expect(&format!("{} could not be loaded as an obj file", file_name));
    let mut indices = dome.indices;
    indices.reverse();

    let positions = dome.vertices.iter().map(|i| i.position).collect();
    let normals = dome.vertices.iter().map(|i| i.normal).collect();
    (positions, normals, indices)
}

pub fn load_cube() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u16>) {
    let positions = vec![
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        // bottom (0, 0, -1)
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        // right (1, 0, 0)
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        // left (-1, 0, 0)
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, -1.0],
        // front (0, 1, 0)
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        // back (0, -1, 0)
        [1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
    ];

    let normals = vec![
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
    ];

    let index_data: Vec<u16> = vec![
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];
    (positions, normals, index_data)
}

pub fn load_plane(size: i8) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u16>) {
    let positions = vec![
        [size as f32, -size as f32, 0.0],
        [size as f32, size as f32, 0.0],
        [-size as f32, -size as f32, 0.0],
        [-size as f32, size as f32, 0.0],
    ];
    /* let positions = vec![
        [-size as f32, -size as f32, 0.0],
        [size as f32, -size as f32, 0.0],
        [size as f32, size as f32, 0.0],
        [-size as f32, size as f32, 0.0],
    ]; */

    let normals = vec![
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];

    let index_data: Vec<u16> = vec![0, 1, 2, 2, 3, 1];
    /* let index_data: Vec<u16> = vec![0, 1, 2, 3]; */
    (positions, normals, index_data)
}

#[rustfmt::skip]
macro_rules! mx_correction {
    () => {
        cgmath::Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        )
    };
}

pub fn generate_light_projection(pos: [f32; 4], fov: f32) -> cgmath::Matrix4<f32> {
    use cgmath::{Deg, EuclideanSpace, Matrix4, PerspectiveFov, Point3, Vector3};
    let mx_view = Matrix4::look_at(
        Point3::new(pos[0], pos[1], pos[2]),
        Point3::origin(),
        Vector3::unit_z(),
    );
    let projection = PerspectiveFov {
        fovy: Deg(fov).into(),
        aspect: 1.0,
        near: 1.0,
        far: 100.0,
    };
    let mx_view_proj = mx_correction!() * Matrix4::from(projection.to_perspective()) * mx_view;
    mx_view_proj
}

pub fn generate_view_matrix() -> cgmath::Matrix4<f32> {
    let mx_view = cgmath::Matrix4::look_at(
        // From this spot
        cgmath::Point3::new(0.05f32, 2.4, -5.0) * 2.0,
        // Look here
        cgmath::Point3::new(0f32, 0.0, 0.0),
        cgmath::Vector3::unit_z(),
    );
    /* let mx_view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(3.0f32, -10.0, 6.0),
            cgmath::Point3::new(0f32, 0.0, 0.0),
            cgmath::Vector3::unit_z(),
        ); */
    mx_view
}

pub fn generate_projection_matrix(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
    let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 100.0);
    mx_correction!() * mx_projection
}

pub fn generate_identity_matrix() -> cgmath::Matrix4<f32> {
    use cgmath::SquareMatrix;
    cgmath::Matrix4::identity()
}

pub fn translate(
    matrix: cgmath::Matrix4<f32>,
    delta_x: f32,
    delta_y: f32,
    delta_z: f32,
) -> cgmath::Matrix4<f32> {
    cgmath::Matrix4::from_translation(cgmath::Vector3::new(delta_x, delta_y, delta_z)) * matrix
}

pub fn rotation_x(matrix: cgmath::Matrix4<f32>, rotate_x: f32) -> cgmath::Matrix4<f32> {
    cgmath::Matrix4::from_angle_x(cgmath::Rad(rotate_x)) * matrix
}

pub fn rotation_y(matrix: cgmath::Matrix4<f32>, rotate_y: f32) -> cgmath::Matrix4<f32> {
    cgmath::Matrix4::from_angle_y(cgmath::Rad(rotate_y)) * matrix
}

pub fn rotation_z(matrix: cgmath::Matrix4<f32>, rotate_z: f32) -> cgmath::Matrix4<f32> {
    cgmath::Matrix4::from_angle_z(cgmath::Rad(rotate_z)) * matrix
}

pub fn rotation(
    matrix: cgmath::Matrix4<f32>,
    rotate_x: f32,
    rotate_y: f32,
    rotate_z: f32,
) -> cgmath::Matrix4<f32> {
    cgmath::Matrix4::from_angle_x(cgmath::Rad(rotate_x))
        * cgmath::Matrix4::from_angle_y(cgmath::Rad(rotate_y))
        * cgmath::Matrix4::from_angle_z(cgmath::Rad(rotate_z))
        * matrix
}

pub fn scale(matrix: cgmath::Matrix4<f32>, scale: f32) -> cgmath::Matrix4<f32> {
    cgmath::Matrix4::from_scale(scale) * matrix
}

pub fn create_texels(size: usize) -> Vec<u8> {
    use std::iter;

    (0..size * size)
        .flat_map(|id| {
            // get high five for recognizing this ;)
            let cx = 3.0 * (id % size) as f32 / (size - 1) as f32 - 2.0;
            let cy = 2.0 * (id / size) as f32 / (size - 1) as f32 - 1.0;
            let (mut x, mut y, mut count) = (cx, cy, 0);
            while count < 0xFF && x * x + y * y < 4.0 {
                let old_x = x;
                x = x * x - y * y + cx;
                y = 2.0 * old_x * y + cy;
                count += 1;
            }
            iter::once(0xFF - (count * 5) as u8)
                .chain(iter::once(0xFF - (count * 15) as u8))
                .chain(iter::once(0xFF - (count * 50) as u8))
                .chain(iter::once(1))
        })
        .collect()
}
