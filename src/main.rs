use bevy::{
    core_pipeline::core_3d::Transparent3d,
    core_pipeline::experimental::taa::{TemporalAntiAliasBundle, TemporalAntiAliasPlugin},
    core_pipeline::{
        bloom::{BloomCompositeMode, BloomPrefilterSettings, BloomSettings},
        tonemapping::Tonemapping,
    },
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    ecs::{
        query::QueryItem,
        system::{lifetimeless::*, SystemParamItem},
    },
    pbr::wireframe::{Wireframe, WireframeConfig, WireframePlugin},
    pbr::CascadeShadowConfigBuilder,
    pbr::{MeshPipeline, MeshPipelineKey, MeshUniform, SetMeshBindGroup, SetMeshViewBindGroup},
    pbr::{
        ScreenSpaceAmbientOcclusionBundle, ScreenSpaceAmbientOcclusionQualityLevel,
        ScreenSpaceAmbientOcclusionSettings,
    },
    prelude::*,
    prelude::*,
    render::camera::TemporalJitter,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::{GpuBufferInfo, MeshVertexBufferLayout},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult,
            RenderPhase, SetItemPipeline, TrackedRenderPass,
        },
        render_resource::*,
        renderer::RenderDevice,
        view::{ExtractedView, NoFrustumCulling},
        Render, RenderApp, RenderSet,
    },
    render::{render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin},
    tasks::{AsyncComputeTaskPool, Task},
};
use bevy_atmosphere::prelude::*;

use bevy_fps_controller::controller::*;
use bevy_rapier3d::prelude::*;
use bevy_render::mesh;
use bytemuck::{Pod, Zeroable};
use noise::{
    utils::{NoiseMapBuilder, PlaneMapBuilder},
    OpenSimplex,
};
use noise::{Blend, Fbm, NoiseFn, Perlin, RidgedMulti};
use rand::Rng;

use bevy_aabb_instancing::{Cuboid, CuboidMaterialId, Cuboids, VertexPullingRenderPlugin};

#[derive(Component)]
struct Block(Vec2, usize);

#[derive(Component)]
struct chunkData(Vec<Vec2>);

#[derive(Component)]
struct Player;

#[derive(Component)]
struct Derender;

#[derive(Component)]
struct FBM(Fbm<Perlin>);

#[derive(Component)]
struct RenderCube(Vec2, usize, usize);

#[derive(Component)]
struct Colliders;


static mut CHUNKINDEX: usize = 0;

fn vec2_to_index(grid_point: Vec2, grid_width: usize) -> usize {
    let index = (grid_point.y as usize) * grid_width + (grid_point.x as usize);
    index
}

fn index_to_coordinates(index: usize, columns: usize) -> (f32, f32) {
    let row = index / columns;
    let column = index % columns;
    (row as f32, column as f32)
}
fn main() {
    App::new()
        .insert_resource(AmbientLight {
            brightness: 5.0,
            ..default()
        })
        .add_plugins((
            DefaultPlugins.set(RenderPlugin {
                wgpu_settings: WgpuSettings {
                    features: WgpuFeatures::POLYGON_MODE_LINE,
                    ..default()
                },
            }),
            CustomMaterialPlugin,
            WireframePlugin,
            AtmospherePlugin,
            VertexPullingRenderPlugin { outlines: true },
            RapierDebugRenderPlugin::default(),
            RapierPhysicsPlugin::<NoUserData>::default()
        ))
        .insert_resource(Msaa::Off)
        .add_plugins(FpsControllerPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (update_chunk, render_chunks))
        .add_plugins(LogDiagnosticsPlugin::default())
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .run();
}

fn spawn_chunk(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    offset: usize,
    playerChunk: Vec2,
    chunkIndexPos: usize,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    block_index: Vec2,
    noise: &Fbm<Perlin>,
) {
    /*
        (fbm.get([(x as f32 / 24.) as f64, (y as f32 / 24.) as f64, 120 as f64])
        * 15.
        + 16.0) as f32

    if ((fbm.get([
                            (x * num_cubes_size + block_index.x) as f64,
                            (y * num_cubes_size + block_index.y) as f64,
                            120 as f64,
                        ]) * 15.
                            + 16.0)
                            / 2.0) as f32 > ((fbm.get([
                                ((x+1.0) * num_cubes_size + block_index.x) as f64,
                                (y * num_cubes_size + block_index.y) as f64,
                                120 as f64,
                            ]) * 15.
                                + 16.0)
                                / 2.0) as f32 {
                                    Color::rgb(0.0,0.7,0.0).as_rgba_f32()
                                } else {
                                    Color::rgb(0.0,1.0,0.0).as_rgba_f32()

                                } //Color::rgba(x * num_cubes_size + block_index.x, y, y * num_cubes_size + block_index.y, 1.0).as_rgba_f32(),
        ((fbm.get([
            ((x * num_cubes + (xOffset * 2.0)) as f32 / 24.) as f64,
            ((y * num_cubes + (yOffset * 2.0)) as f32 / 24.) as f64,
            120 as f64,
        ]) * 15.
            + 16.0)
            / 2.0) as f32;


        let num_cubes = 32;
        let num_cubes_half = (num_cubes / 2 )as f32;
        let num_cubes_size = 32.0*32.0;
        commands.spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Cube { size: 64.0 })),
                ..Default::default()
            },
            //SpatialBundle::INHERITED_IDENTITY,
            InstanceMaterialData(
                (1..=num_cubes as i32)
                    .flat_map(|x| {
                        (1..=num_cubes as i32)
                            .map(move |y| (x as f32 / num_cubes_half, y as f32 / num_cubes_half))
                    })
                    .map(|(x, y)| InstanceData {
                        position: Vec3::new(
                            x * num_cubes_size + block_index.x,
                            ((fbm.get([
                                (x * num_cubes_size + block_index.x) as f64 / 24.0,
                                (y * num_cubes_size + block_index.y) as f64 / 24.0,
                                120 as f64,
                            ]) * 15.)) as f32,
                             y * num_cubes_size + block_index.y,
                        ),
                        scale: 2.0,
                        color: color.as_rgba_f32()
                    })
                    .collect(),
            ),


            // NOTE: Frustum culling is done based on the Aabb of the Mesh and the GlobalTransform.
            // As the cube is at the origin, if its Aabb moves outside the view frustum, all the
            // instanced cubes will be culled.
            // The InstanceMaterialData contains the 'GlobalTransform' information for this custom
            // instancing, and that is not taken into account with the built-in frustum culling.
            // We must disable the built-in frustum culling by adding the `NoFrustumCulling` marker
            // component to avoid incorrect culling.
            NoFrustumCulling,
            Block(block_index, offset),
            Wireframe,
        ));

        for index in 0..32 * 32 {
            let (mut x, mut z) = index_to_coordinates(index, 32);

            x += 1.0;
            z += 1.0;


        }


        commands.spawn((

            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Cube { size: 10.0 })),
                transform: Transform::from_xyz(
                    block_index.x,
                    yas,
                    block_index.y,
                ),
                material:  materials.add(color.into()),
                ..default()
            },
            Block(block_index, offset),

            //Collider::cuboid(2.0, 0.5, 2.0),
        ));


    let extent: f64 = 64.0 * 64.0;
    let intensity = 0.3;
    let width: usize = 32;
    let depth: usize = 32;

    // Create noisemap
    fbm.frequency = 1.0;
    fbm.lacunarity = 3.0;
    fbm.octaves = 6;

    let vertices_count: usize = (width + 1) * (depth + 1);
    let triangle_count: usize = width * depth * 2 * 3;

    // Cast
    let (width_u32, depth_u32) = (width as u32, depth as u32);
    let (width_f32, depth_f32) = (width as f32, depth as f32);
    let extent_f32 = extent as f32;

    // Defining vertices.
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(vertices_count);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(vertices_count);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(vertices_count);
    //(fbm.get([((d as f32 + block_index.x) / 24.) as f64, ((w as f32 + block_index.x) / 24.) as f64, 123 as f64]) * 15.) as f32
    let mut rng = rand::thread_rng();
    let perlin = Perlin::default();
    let ridged = RidgedMulti::<Perlin>::default();
    let mut fbm: Fbm<Perlin> = Fbm::<Perlin>::default();
    fbm.octaves = 5;
    fbm.frequency = 5.0;

    let mut heightSwitch = true; // Initialize your variable
    for d in 0..=width {
        let height = if heightSwitch{
            60
        } else {
            1
        };


        if d % 4 == 0 {
            heightSwitch = !heightSwitch;

        }
        //rng.gen_range(0..120);
        for w in 0..=depth {
            let (mut w_f32, mut d_f32) = (w as f32, d as f32);

            let wIndex = if w == 0 {
                w
            } else {
                w - 1
            };




            let x = (w_f32 - width_f32 / 2.) * extent_f32 / width_f32;
            let z = (d_f32 - depth_f32 / 2.) * extent_f32 / depth_f32;

            let pos = [
                x,  //height_map[bZ as usize - 1][bX as usize - 1] as f32,
                height as f32,
                /*fbm.get([
                    (x + block_index.x) as f64 / 24.0,
                    (z + block_index.y) as f64 / 24.0,
                    123.0 as f64,
                ]) as f32 * 15.0,//map.get_value((x + block_index.x) as usize, (z + block_index.y) as usize) as f32,//fbm.get([w as f64, d as f64, 123.0]) as f32 * 100.0,
                */

                z,
            ];


            positions.push(pos);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([w_f32 / width_f32, d_f32 / depth_f32]);
        }
    }

    // Defining triangles.
    let mut triangles: Vec<u32> = Vec::with_capacity(triangle_count);

    for d in 0..depth_u32 {
        for w in 0..width_u32 {
            // First tringle
            triangles.push((d * (width_u32 + 1)) + w);
            triangles.push(((d + 1) * (width_u32 + 1)) + w);
            triangles.push(((d + 1) * (width_u32 + 1)) + w + 1);
            // Second triangle
            triangles.push((d * (width_u32 + 1)) + w);
            triangles.push(((d + 1) * (width_u32 + 1)) + w + 1);
            triangles.push((d * (width_u32 + 1)) + w + 1);

        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    mesh.set_indices(Some(bevy::render::mesh::Indices::U32(triangles)));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);

    commands
        .spawn((
            PbrBundle {
                mesh: meshes.add(mesh),
                material: materials.add(StandardMaterial {
                    base_color: Color::INDIGO,
                    perceptual_roughness: 1.0,
                    ..default()
                }),
                transform: Transform::from_xyz(block_index.x, 0.0, block_index.y),
                ..default()
            },
            Block(block_index, offset),
        ))
        .insert(Wireframe);
    */
        let mut instances: Vec<Cuboid> = Vec::with_capacity(32 * 32);

        for i in 0..32 * 32 {
            let (mut x, mut y) = index_to_coordinates(i, 32);

            let xPos = x * 128.0;
            let yPos = y * 128.0;
            let offset = Vec2::new(f32::floor(block_index.x / 64.0), f32::floor(block_index.y / 64.0));

            let val = (noise.get([
                ((x + offset.x) / 2048.0) as f64,
                ((y + offset.y) / 2048.0) as f64,
                123 as f64,
            ]) *64.0);

//         4096.0) 2304  println!("{}", val);

            let c = Vec3::new(xPos, val.floor() as f32 * 128.0, yPos);

            let size = Vec3::new(64.0, 64.0, 64.0);

            let min = c - size; //c + glam::vec3(1.0, 1.0, 1.0);//glam::f32::vec3(1.0, 1.0, 1.0);
            let max = c + size; //Vec3::new(1.0, 1.0, 1.0);

            let cuboid = Cuboid::new(min, max, Color::GREEN.as_rgba_u32());
            instances.push(cuboid);

        }


    let cuboids = Cuboids::new(instances);
    let aabb = cuboids.aabb();
    commands
        .spawn((
            PbrBundle {
                transform: Transform::from_xyz(block_index.x * 2.0, 0.0, block_index.y * 2.0),
                material: materials.add(Color::GREEN.into()),
                ..Default::default()
            },
            Block(block_index, offset),
        ))
        .insert((cuboids, aabb, CuboidMaterialId(0)));
}

fn update_chunk(
    mut chunkQuery: Query<
        (Entity, &mut Transform, &Block),
        (With<Block>, Without<Player>, Without<Derender>),
    >,
    mut data: Query<&mut chunkData>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,

    mut player: Query<&mut Transform, With<Player>>,
    mut fbm: Query<&FBM>,
) {
    let mut chunkD = &mut data.single_mut().0;
    let mut chunk_index = 0;
    let p = player.single_mut();

    let point1: (f32, f32) = (p.translation.x, p.translation.z); //(0.0,0.0);

    let cell_size = 16384.0 / 4.0; // The size of each chunk cell

    let rounded_x = f32::floor(point1.0 / cell_size);
    let rounded_y = f32::floor(point1.1 / cell_size);

    let playerPosChunk = vec2_to_index(Vec2::new(rounded_x, rounded_y), 30);

    //println!("{}", playerPosChunk);

    //vec2_to_index(Vec2::new(rounded_x, rounded_y), 15);
    let colors = [
        Color::RED,
        Color::GREEN,
        Color::BLACK,
        Color::ORANGE,
        Color::BLUE,
        Color::PURPLE,
        Color::YELLOW,
        Color::YELLOW_GREEN,
        Color::ORANGE_RED,
    ];

    let mut rng = rand::thread_rng();

    for i in 0..324 {
        let (chunkIndexPosx, chunkIndexPosxy) = index_to_coordinates(i, 18);

        let (mut xOffset, mut yOffset) = index_to_coordinates(i, 18);
        // - 10859.
        let block_index = vec2_to_index(Vec2::new(xOffset + rounded_x, yOffset + rounded_y), 18);
        let chunkPos = Vec2::new(
            (xOffset + rounded_x) * 2048. - 14656.0,
            (yOffset + rounded_y) * 2048. - 14656.0,
        );

        let noise = fbm.single();

        if !(chunkQuery.iter().any(|(e, p, b)| chunkPos == b.0)) {
            //colors[rng.gen_range(0..8)]let color: Color = colors[rng.gen_range(0..8)];

            //println!("{}: {}", chunk_index, chunkD.contains(&chunkPos));

            spawn_chunk(
                &mut commands,
                &mut meshes,
                i,
                Vec2::new(rounded_x, rounded_y),
                i + playerPosChunk,
                &mut materials,
                chunkPos,
                &noise.0,
            );
//            commands.spawn(RenderCube(chunkPos, i, i + playerPosChunk));

            //;

            if chunkD.len() >= 324 {
                chunkD.remove(0);
            }

            chunkD.push(chunkPos); //vec2_to_index(Vec2::new(xOffset + rounded_x, yOffset + rounded_y), 15);
            chunk_index += 1;
        }
    }

    chunkQuery.for_each(|(mut e, mut pos, b)| {
        if !chunkD.contains(&b.0) {
            commands.entity(e).insert(Derender);
        }

    });
}

fn render_chunks(
    mut query: Query<Entity, (With<Derender>, With<Block>)>,
    mut query1: Query<(&Block, &mut Transform), (Without<Derender>, Without<Player>)>,
    mut data: Query<&mut chunkData>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut player: Query<&mut Transform, With<Player>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut fbm: Query<&FBM>,
    mut colliders: Query<&Collider>

) {
    let p = player.single_mut();

    let point1: (f32, f32) = (p.translation.x, p.translation.z); //(0.0,0.0);
    let cell_size = 16384.0 / 4.0; // The size of each chunk cell

    let rounded_x = f32::floor(point1.0 / cell_size);
    let rounded_y = f32::floor(point1.1 / cell_size);
    let (mut xOffset, mut yOffset) = index_to_coordinates(158, 18);

    let chunkPos = Vec2::new(
        (xOffset + rounded_x) * 2048. - 14656.0,
        (yOffset + rounded_y) * 2048. - 14656.0,
    );

    query.for_each(|(e)| {
        commands.entity(e).despawn();
    });



    /*
    let noise = fbm.single();
    let p = player.single_mut();

    let point1: (f32, f32) = (p.translation.x, p.translation.z); //(0.0,0.0);

    let cell_size = 16384.0 / 4.0; // The size of each chunk cell


    let rounded_x = f32::floor(point1.0 / cell_size);
    let rounded_y = f32::floor(point1.1 / cell_size);

    let playerPosChunk = vec2_to_index(Vec2::new(rounded_x, rounded_y), 30);
    query1.for_each(|(i, e)| {
        spawn_chunk(
            &mut commands,
            &mut meshes,
            e.1,
            Vec2::new(rounded_x, rounded_y),
            e.2,
            &mut materials,
            e.0,
            &noise.0,
        );

        commands.entity(i).despawn();
    });



    let mut chunkD = &mut data.single_mut().0;

    if unsafe { CHUNKINDEX } >= query.iter().len() {
        unsafe { CHUNKINDEX = 0 };
    }

    let element = query.iter().nth(unsafe { CHUNKINDEX });

    element.as_ref().map(|e| {

        commands.entity(*e).despawn();

        /*
        if !chunkD.contains(&block.0) {
            commands.entity(*e).despawn();
            println!("{}", &block.0);

        }*/
    });

    unsafe { CHUNKINDEX += 1 };

    */
}

fn setup(mut commands: Commands, _meshes: ResMut<Assets<Mesh>>) {
    //spawn_chunk(&mut commands, &mut meshes);

    let mut base_data = Vec::<Vec2>::with_capacity(324);
    let mut base_data2 = Vec::<Vec2>::with_capacity(324);

    for _i in 0..324 {
        base_data.push(Vec2::MAX);
        base_data2.push(Vec2::MAX);

    }

    for _col in 0..9 {
        commands.spawn((
            Collider::cuboid(1.0, 1.0, 1.0),
            TransformBundle::from(Transform::from_xyz(0.0, 0.0, 0.0)),
            Colliders,
        ));
    }

    let mut rng = rand::thread_rng();

    let mut fbm = Fbm::<Perlin>::new(rng.gen_range(1000..10000));
	fbm.octaves = 6;
    fbm.frequency = 2.0;
    commands.spawn(FBM(fbm));
    commands.spawn(chunkData(base_data));
    commands.spawn((
        (
            Collider::capsule(Vec3::Y * 0.5, Vec3::Y * 1.5, 0.5),
            Friction {
                coefficient: 0.0,
                combine_rule: CoefficientCombineRule::Min,
            },
            Restitution {
                coefficient: 0.0,
                combine_rule: CoefficientCombineRule::Min,
            },
            ActiveEvents::COLLISION_EVENTS,
            Velocity::zero(),
            RigidBody::Dynamic,
            Sleeping::disabled(),
            LockedAxes::ROTATION_LOCKED,
            AdditionalMassProperties::Mass(1.0),
            GravityScale(0.0),
            Ccd { enabled: true }, // Prevent clipping when going fast
            TransformBundle::from_transform(Transform::from_xyz(0.0, 15.0, 0.0)),
            LogicalPlayer(0),
            FpsControllerInput {
                pitch: -1.0 / 12.0,
                yaw: 1.0 * 5.0 / 8.0,
                ..default()
            },
            FpsController {
                gravity: 0.0,

                //jump_speed: 8.0,
                //friction: 0.001,
                fly_speed: 6000.0,
                fast_fly_speed: 480.0,
                //max_air_speed: 10000000000.0,
                //walk_speed: 100000000000000000000000.0,
                ..default()
            },
        ),
        Player,
    ));

    commands
        .spawn((Camera3dBundle {
            camera: Camera {
                hdr: true,
                ..default()
            },
            tonemapping: Tonemapping::TonyMcMapface,

            ..default()
        },
        FogSettings {
            color: Color::rgba(0.05, 0.05, 0.05, 1.0),
            falloff: FogFalloff::Linear {
                start: 5.0,
                end: 20.0,
            },
            ..default()
        },
        
    ),
    
    )
        //.insert(ScreenSpaceAmbientOcclusionBundle::default())
        //.insert(TemporalAntiAliasBundle::default())
        .insert(RenderPlayer(0));
    

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform {
            translation: Vec3::new(0.0, 15.0, 0.0),
            rotation: Quat::from_rotation_x(-3.14 / 4.),
            ..default()
        },
        // The default cascade config is designed to handle large scenes.
        // As this example has a much smaller world, we can tighten the shadow
        // bounds for better visual quality.
        cascade_shadow_config: CascadeShadowConfigBuilder {
            first_cascade_far_bound: 4.0,
            ..default()
        }
        .into(),
        ..default()
    });
}

#[derive(Component, Deref)]
struct InstanceMaterialData(Vec<InstanceData>);

impl ExtractComponent for InstanceMaterialData {
    type Query = &'static InstanceMaterialData;
    type Filter = ();
    type Out = Self;

    fn extract_component(item: QueryItem<'_, Self::Query>) -> Option<Self> {
        Some(InstanceMaterialData(item.0.clone()))
    }
}

pub struct CustomMaterialPlugin;

impl Plugin for CustomMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<InstanceMaterialData>::default());
        app.sub_app_mut(RenderApp)
            .add_render_command::<Transparent3d, DrawCustom>()
            .init_resource::<SpecializedMeshPipelines<CustomPipeline>>()
            .add_systems(
                Render,
                (
                    queue_custom.in_set(RenderSet::Queue),
                    prepare_instance_buffers.in_set(RenderSet::Prepare),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp).init_resource::<CustomPipeline>();
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct InstanceData {
    position: Vec3,
    scale: f32,
    color: [f32; 4],
}

#[allow(clippy::too_many_arguments)]
fn queue_custom(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<CustomPipeline>,
    msaa: Res<Msaa>,
    mut pipelines: ResMut<SpecializedMeshPipelines<CustomPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<Mesh>>,
    material_meshes: Query<(Entity, &MeshUniform, &Handle<Mesh>), With<InstanceMaterialData>>,
    mut views: Query<(&ExtractedView, &mut RenderPhase<Transparent3d>)>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawCustom>();

    let msaa_key = MeshPipelineKey::from_msaa_samples(msaa.samples());

    for (view, mut transparent_phase) in &mut views {
        let view_key = msaa_key | MeshPipelineKey::from_hdr(view.hdr);
        let rangefinder = view.rangefinder3d();
        for (entity, mesh_uniform, mesh_handle) in &material_meshes {
            if let Some(mesh) = meshes.get(mesh_handle) {
                let key =
                    view_key | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                let pipeline = pipelines
                    .specialize(&pipeline_cache, &custom_pipeline, key, &mesh.layout)
                    .unwrap();

                transparent_phase.add(Transparent3d {
                    entity,
                    pipeline,
                    draw_function: draw_custom,
                    distance: rangefinder.distance(&mesh_uniform.transform),
                });
            }
        }
    }
}

#[derive(Component)]
pub struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
}

fn prepare_instance_buffers(
    mut commands: Commands,
    query: Query<(Entity, &InstanceMaterialData)>,
    render_device: Res<RenderDevice>,
) {
    for (entity, instance_data) in &query {
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("instance data buffer"),
            contents: bytemuck::cast_slice(instance_data.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });
        commands.entity(entity).insert(InstanceBuffer {
            buffer,
            length: instance_data.len(),
        });
    }
}

#[derive(Resource)]
pub struct CustomPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
}

impl FromWorld for CustomPipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let shader = asset_server.load("shaders/instancing.wgsl");

        let mesh_pipeline = world.resource::<MeshPipeline>();

        CustomPipeline {
            shader,
            mesh_pipeline: mesh_pipeline.clone(),
        }
    }
}

impl SpecializedMeshPipeline for CustomPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;

        // meshes typically live in bind group 2. because we are using bindgroup 1
        // we need to add MESH_BINDGROUP_1 shader def so that the bindings are correctly
        // linked in the shader
        descriptor
            .vertex
            .shader_defs
            .push("MESH_BINDGROUP_1".into());

        descriptor.vertex.shader = self.shader.clone();
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 3, // shader locations 0-2 are taken up by Position, Normal and UV attributes
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32x4.size(),
                    shader_location: 4,
                },
            ],
        });
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        Ok(descriptor)
    }
}

type DrawCustom = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    DrawMeshInstanced,
);

pub struct DrawMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawMeshInstanced {
    type Param = SRes<RenderAssets<Mesh>>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = (Read<Handle<Mesh>>, Read<InstanceBuffer>);

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        (mesh_handle, instance_buffer): (&'w Handle<Mesh>, &'w InstanceBuffer),
        meshes: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_mesh = match meshes.into_inner().get(mesh_handle) {
            Some(gpu_mesh) => gpu_mesh,
            None => return RenderCommandResult::Failure,
        };

        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            GpuBufferInfo::Indexed {
                buffer,
                index_format,
                count,
            } => {
                pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                pass.draw_indexed(0..*count, 0, 0..instance_buffer.length as u32);
            }
            GpuBufferInfo::NonIndexed => {
                pass.draw(0..gpu_mesh.vertex_count, 0..instance_buffer.length as u32);
            }
        }
        RenderCommandResult::Success
    }
}
