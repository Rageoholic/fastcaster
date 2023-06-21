use std::{sync::mpsc::channel, thread};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::{IntoParallelIterator, ParallelExtend, ParallelIterator};
use softbuffer::GraphicsContext;
use vek::{Lerp, Ray, Rgb, Vec3};
use winit::{
    dpi::PhysicalSize,
    event::{Event, StartCause, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder},
    window::WindowBuilder,
};

const WIDTH: usize = 400;
const HEIGHT: usize = 300;

#[derive(Debug, Clone, Copy)]
struct Pixel {
    red: u8,
    green: u8,
    blue: u8,
}

#[derive(Clone, Copy)]
enum MaterialType {
    Lambertian,
    Metal(f32),
}

#[derive(Clone, Copy)]
struct Material {
    color: Rgb<f32>,
    mat_type: MaterialType,
}

#[derive(Clone, Copy)]
struct Sphere {
    origin: vek::Vec3<f32>,
    radius: f32,
    material: Material,
}
struct World<'a> {
    spheres: &'a [Sphere],
}

impl Pixel {
    /// Turn pixel into an RGB u32. R high, top padding
    fn to_u32(self) -> u32 {
        let r_channel = (self.red as u32) << 16;
        let g_channel = (self.green as u32) << 8;
        let b_channel = (self.blue as u32) << 0;
        r_channel | g_channel | b_channel
    }

    /// Create a pixel from an RGB vec in the [0,1] range
    fn from_vek_color(v: Rgb<f32>) -> Self {
        Self {
            red: (v.r.clamp(0.0, 1.0) * 255.99) as u8,
            green: (v.g.clamp(0.0, 1.0) * 255.99) as u8,
            blue: (v.b.clamp(0.0, 1.0) * 255.99) as u8,
        }
    }
}

#[allow(dead_code)]
fn visualize_normal(normal: Vec3<f32>) -> Rgb<f32> {
    (normal / 2.0 + 0.5).into()
}

#[derive(Debug, Copy, Clone)]
struct HitRecord {
    intersection_point: Vec3<f32>,
    surface_normal: Vec3<f32>,
    distance: f32,
}

const SHADOW_ACNE_FUDGE_CONSTANT: f32 = 0.001;

trait RandVec: Rng {
    fn rand_vec3(&mut self) -> Vec3<f32> {
        Vec3::new(
            self.gen_range(-1.0..1.0),
            self.gen_range(-1.0..1.0),
            self.gen_range(-1.0..1.0),
        )
    }

    fn rand_vec3_in_unit_sphere(&mut self) -> Vec3<f32> {
        let mut v;
        loop {
            v = self.rand_vec3();
            if v.magnitude_squared() < 1.0 {
                break;
            }
        }
        v
    }

    fn rand_unit_vec3(&mut self) -> Vec3<f32> {
        self.rand_vec3_in_unit_sphere().normalized()
    }

    fn rand_in_hemisphere(&mut self, normal: Vec3<f32>) -> Vec3<f32> {
        let in_unit_sphere = self.rand_vec3_in_unit_sphere();
        if in_unit_sphere.dot(normal) > 0.0 {
            in_unit_sphere
        } else {
            -in_unit_sphere
        }
    }
}

impl<T: Rng> RandVec for T {}

fn hit_sphere(ray: Ray<f32>, sphere: Sphere) -> Option<HitRecord> {
    let oc = ray.origin - sphere.origin;
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * oc.dot(ray.direction);
    let c = oc.dot(oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant > 0.0 {
        let neg_distance = (-b - discriminant.sqrt()) / (2.0 * a);
        let pos_distance = (-b + discriminant.sqrt()) / (2.0 * a);
        let distance = if neg_distance > SHADOW_ACNE_FUDGE_CONSTANT {
            neg_distance
        } else if pos_distance > SHADOW_ACNE_FUDGE_CONSTANT {
            pos_distance
        } else {
            return None;
        };
        let intersection_point = ray.origin + ray.direction * distance;
        let surface_normal = (intersection_point - sphere.origin).normalized();
        Some(HitRecord {
            intersection_point,
            surface_normal,
            distance,
        })
    } else {
        None
    }
}
const MAX_DEPTH: usize = 100;

fn reflected(v: Vec3<f32>, n: Vec3<f32>) -> Vec3<f32> {
    return v - 2.0 * v.dot(n) * n;
}

fn ray_cast(mut ray: Ray<f32>, world: &World, rng: &mut impl rand::Rng) -> Rgb<f32> {
    let t = 1.0 - 0.5 * (ray.direction.y + 1.0);
    let background_color = Lerp::lerp(Rgb::broadcast(1.0), Rgb::new(0.5, 0.7, 1.0), 1.0 - t);
    let mut color = Rgb::broadcast(1.0);
    for _ in 0..MAX_DEPTH {
        let mut min_hit_record: Option<(HitRecord, Material)> = None;
        for sphere in world.spheres {
            if let Some(hit_record) = hit_sphere(ray, *sphere) {
                min_hit_record = min_hit_record
                    .map(|(mhr, color)| {
                        if mhr.distance < hit_record.distance {
                            (mhr, color)
                        } else {
                            (hit_record, sphere.material)
                        }
                    })
                    .or(Some((hit_record, sphere.material)));
            }
        }

        if let Some((hit_record, hit_material)) = min_hit_record {
            color *= hit_material.color;
            match hit_material.mat_type {
                MaterialType::Lambertian => {
                    let random = rng.rand_unit_vec3();
                    ray = Ray::new(
                        hit_record.intersection_point,
                        (hit_record.surface_normal + random).normalized(),
                    );
                }
                MaterialType::Metal(fuzz) => {
                    let reflected = reflected(ray.direction, hit_record.surface_normal)
                        + fuzz * rng.rand_vec3_in_unit_sphere();
                    if reflected.dot(hit_record.surface_normal) > 0.0 {
                        ray = Ray::new(hit_record.intersection_point, reflected.normalized())
                    } else {
                        color *= Rgb::broadcast(0.0);
                        break;
                    }
                }
            }
        } else {
            color *= background_color;
            break;
        }
    }
    color
}

#[derive(Debug)]
struct ThreadRedrawCompleteEvent(Vec<u32>);

fn draw(draw_size: PhysicalSize<u32>, world: &World) -> Vec<u32> {
    let (width, height) = (draw_size.width as usize, draw_size.height as usize);
    let aspect_ratio = width as f32 / height as f32;
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;

    let origin = Vec3::broadcast(0.0);
    let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
    let vertical = Vec3::new(0.0, -viewport_height, 0.0);

    let upper_left_corner =
        origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);
    let sample_count = 4;

    let mut buffer: Vec<u32> = Vec::with_capacity(width * height);
    let seed = rand::rngs::OsRng.gen();

    buffer.par_extend((0..width * height).into_par_iter().map(|i| {
        let x = i % width;
        let y = i / width;
        let mut pixel_color = Rgb::broadcast(0.0);
        let mut rng = StdRng::seed_from_u64(u64::wrapping_add(seed, i as u64));
        for _ in 0..sample_count {
            let v = (y as f32 + rng.gen::<f32>()) / (height as f32 - 1.0);
            let u = (x as f32 + rng.gen::<f32>()) / (width as f32 - 1.0);

            let normalized_direction =
                (upper_left_corner + u * horizontal + v * vertical - origin).normalized();
            if !normalized_direction.is_normalized() {
                eprintln!("non normal vector");
            }
            let ray = Ray::new(origin, normalized_direction);

            pixel_color += ray_cast(ray, world, &mut rng);
        }
        let pixel_color = pixel_color / sample_count as f32;

        let pixel_color = pixel_color.map(|f| f.sqrt());

        let pixel_bits = Pixel::from_vek_color(pixel_color);
        pixel_bits.to_u32()
    }));

    buffer
}

fn main() {
    let event_loop = EventLoopBuilder::<ThreadRedrawCompleteEvent>::with_user_event().build();
    let window = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(WIDTH as f32, HEIGHT as f32))
        .build(&event_loop)
        .unwrap();

    let mut window_size = PhysicalSize::new(WIDTH as u32, HEIGHT as u32);

    let mut buffer = vec![0; WIDTH * HEIGHT];

    let event_loop_proxy = event_loop.create_proxy();

    let (sender, receiver) = channel::<PhysicalSize<u32>>();

    let mut graphics_context = unsafe { GraphicsContext::new(&window, &window).unwrap() };

    sender.send(window.inner_size()).unwrap();

    let _thread = thread::spawn(move || loop {
        let draw_size = receiver.recv().unwrap();
        let world = World {
            spheres: &[
                Sphere {
                    origin: Vec3::new(0.0, 0.0, -1.0),
                    radius: 0.5,
                    material: Material {
                        color: Rgb {
                            r: 0.7,
                            g: 0.3,
                            b: 0.3,
                        },
                        mat_type: MaterialType::Lambertian,
                    },
                },
                Sphere {
                    origin: Vec3::new(0.0, -100.5, -1.0),
                    radius: 100.0,
                    material: Material {
                        color: Rgb::new(0.8, 0.8, 0.3),
                        mat_type: MaterialType::Lambertian,
                    },
                },
                Sphere {
                    origin: Vec3::new(-1.0, 0.0, -1.0),
                    radius: 0.5,
                    material: Material {
                        color: Rgb::new(0.8, 0.8, 0.8),
                        mat_type: MaterialType::Metal(0.3),
                    },
                },
                Sphere {
                    origin: Vec3 {
                        x: 1.0,
                        y: 0.0,
                        z: -1.0,
                    },
                    radius: 0.5,
                    material: Material {
                        color: Rgb::new(0.8, 0.6, 0.2),
                        mat_type: MaterialType::Metal(1.0),
                    },
                },
            ],
        };
        event_loop_proxy
            .send_event(ThreadRedrawCompleteEvent(draw(draw_size, &world)))
            .unwrap();
    });

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            window_id: _,
            event,
        } => match event {
            WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::Exit;
            }
            WindowEvent::Resized(new_size) => {
                if window_size != new_size {
                    buffer = vec![0; (new_size.width * new_size.height) as usize];
                    sender.send(new_size).unwrap();
                    println!("resized from {:?} to {:?}", window_size, new_size);
                    window_size = new_size;
                }
            }
            _ => {}
        },
        Event::UserEvent(ThreadRedrawCompleteEvent(new_buf)) => {
            if buffer.len() == new_buf.len() {
                buffer = new_buf;
                window.request_redraw();
                println!("New display!");
            }
        }
        Event::RedrawRequested(_win_id) => {
            let (width, height) = {
                let inner_size = window.inner_size();
                (inner_size.width, inner_size.height)
            };
            graphics_context.set_buffer(&buffer, width as u16, height as u16);
        }
        Event::NewEvents(StartCause::Init) => *control_flow = ControlFlow::Wait,
        _ => {}
    })
}
