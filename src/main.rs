use std::{sync::mpsc::channel, thread};

use softbuffer::GraphicsContext;
use winit::{
    dpi::PhysicalSize,
    event::{Event, StartCause, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder},
    window::WindowBuilder,
};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;

#[derive(Debug, Clone, Copy)]
struct Pixel {
    red: u8,
    green: u8,
    blue: u8,
}

impl Pixel {
    fn to_u32(self) -> u32 {
        let r_channel = (self.red as u32) << 16;
        let g_channel = (self.green as u32) << 8;
        let b_channel = (self.blue as u32) << 0;
        r_channel | g_channel | b_channel
    }
}

#[derive(Debug)]
struct ThreadRedrew(Vec<u32>);

fn draw(draw_size: PhysicalSize<u32>) -> Vec<u32> {
    let (width, height) = (draw_size.width as usize, draw_size.height as usize);
    let mut buffer = Vec::with_capacity(width * height);
    for pixel_number in 0..(width * height) {
        let (x, y) = (pixel_number % width, pixel_number / width);
        let r = 0;
        let g = x % 256;
        let b = y % 256;
        buffer.push(
            Pixel {
                red: r,
                green: g as u8,
                blue: b as u8,
            }
            .to_u32(),
        );
    }

    buffer
}

fn main() {
    let event_loop = EventLoopBuilder::<ThreadRedrew>::with_user_event().build();
    let window = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(WIDTH as f32, HEIGHT as f32))
        .build(&event_loop)
        .unwrap();

    let mut buffer = vec![0; WIDTH * HEIGHT];

    let event_loop_proxy = event_loop.create_proxy();

    let (sender, receiver) = channel::<PhysicalSize<u32>>();

    let mut graphics_context = unsafe { GraphicsContext::new(&window, &window).unwrap() };

    sender.send(window.inner_size()).unwrap();

    let _thread = thread::spawn(move || loop {
        let draw_size = receiver.recv().unwrap();
        event_loop_proxy
            .send_event(ThreadRedrew(draw(draw_size)))
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
                buffer = vec![0; (new_size.width * new_size.height) as usize];
                sender.send(new_size).unwrap();
            }
            _ => {}
        },
        Event::UserEvent(ThreadRedrew(new_buf)) => {
            if buffer.len() == new_buf.len() {
                buffer = new_buf;
            }
            window.request_redraw();
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
