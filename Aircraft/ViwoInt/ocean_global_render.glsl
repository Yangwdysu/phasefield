#version 450 core

####include trigonometry.glsl
####include water_lighting.glsl
####include ocean_global_render_data.glsl
####include ocean_global_common.glsl
#define PolutionNum 4
#ifdef GLSL_VERTEX

layout(location = 0) in dvec2 lonlat;
layout(location = 2) in uint neighbourSplit;

out dvec2 lonlatCtrl;
out float outerLevel;
out vec2 screenCoord;

vec2 lonlat2screen(const dvec2 lonlat)
{
    const dvec3 world = lonlat2world(lonlat, EarthRadius);
    const dvec4 clip = projViewMat * dvec4(world, 1.0);
    const vec2 normalized_screen = vec2(clip.xy / clip.w) * 0.5 + 0.5;
    return normalized_screen * screenSize;
}

void main()
{
    lonlatCtrl = lonlat;
    outerLevel = baseSplit * neighbourSplit;
    screenCoord = lonlat2screen(lonlat);
}

#endif

#ifdef GLSL_TESS_CONTROL

in dvec2 lonlatCtrl[];
in float outerLevel[];
in vec2 screenCoord[];

layout(vertices = 4) out;

out dvec2 lonlatEval[];

patch out float side_len[4];

void main()
{
    const int idx = gl_InvocationID;
    
    lonlatEval       [gl_InvocationID] = lonlatCtrl  [gl_InvocationID];
    gl_TessLevelOuter[gl_InvocationID] = outerLevel  [gl_InvocationID];
    
    const int next_idx = (idx + 1) % 4;
    const vec2 start_screen = screenCoord[idx];
    const vec2 end_screen = screenCoord[next_idx];
    side_len[idx] = distance(start_screen, end_screen);
    barrier();
    
    if(idx < 2)
    {
        const float desired_grid_len = 32;
        
        const float longest_side = max(side_len[idx], side_len[idx + 2]);
        
        const float desired_split = longest_side / desired_grid_len;
        const float real_split = min(desired_split, baseSplit);

		gl_TessLevelInner[idx] = baseSplit;
		//gl_TessLevelInner[idx] = real_split;
    }
}

#endif

#ifdef GLSL_TESS_EVALUATION

layout(quads, equal_spacing, ccw) in;

in dvec2 lonlatEval[];

out vec4 frag_clipCoord;
out vec3 local_pos;

out vec2 oceanTexCoord;
out vec2 oceanBoatGrad;

out vec3 polution_vec[PolutionNum];
out vec3 displacementValue;
const dvec3 polution_world[PolutionNum] = dvec3[](
	lonlat2world(dvec2(113.275499674963, 22.207336655410) / 180.0 * 3.1415926, EarthRadius),
	lonlat2world(dvec2(113.729448734649, 22.812862337093) / 180.0 * 3.1415926, EarthRadius),
	lonlat2world(dvec2(114.022414204110, 22.434636826536) / 180.0 * 3.1415926, EarthRadius),
	lonlat2world(dvec2(114.405107634589, 22.613673494302) / 180.0 * 3.1415926, EarthRadius));
dvec2 mix2d(dvec2 v0, dvec2 v1, dvec2 v2, dvec2 v3, vec2 f)
{
    return mix(mix(v0, v1, f.x), mix(v3, v2, f.x), f.y);
}

vec2 mix2d(vec2 v0, vec2 v1, vec2 v2, vec2 v3, vec2 f)
{
    return mix(mix(v0, v1, f.x), mix(v3, v2, f.x), f.y);
}

void main()
{
    const dvec2 mid = mix2d(lonlatEval[0], lonlatEval[1], lonlatEval[2], lonlatEval[3], gl_TessCoord.xy);
    const dvec3 normalized_pos = lonlat2world(mid, 1.0);
    const dvec3 world_pos_sphere = normalized_pos * EarthRadius;
    const vec3 local_pos_planar = vec3(pos2local(normalized_pos, east, north, surface_up) * EarthRadius, 0.0);
	
    vec2 uv_local;
	vec2 boat_grad;
    const vec3 displacement = calcOceanCoords(local_pos_planar, uv_local, boat_grad);
    
    const dvec3 world_pos = world_pos_sphere + (east * displacement.x + north * displacement.y + surface_up * displacement.z);
    for(int i=0; i<PolutionNum; i++) 
		polution_vec[i] = vec3(world_pos - polution_world[i]);
    local_pos = local_pos_planar + displacement.xyz;
    oceanTexCoord = uv_local;
	oceanBoatGrad = boat_grad;
	displacementValue = displacement;
    gl_Position = vec4(projViewMat * dvec4(world_pos, 1.0));
    frag_clipCoord = gl_Position;
}

#endif

#ifdef GLSL_FRAGMENT

in vec4 frag_clipCoord;
in vec3 local_pos;

in vec2 oceanTexCoord;
in vec2 oceanBoatGrad;
in vec3 polution_vec[PolutionNum];
in vec3 displacementValue;
layout(early_fragment_tests) in;
layout(location = 0) out vec4 color_out;
layout(location = 1) out vec4 normal_out;
layout(location = 2) out vec3 emit_out;
layout(location = 3) out uint material_out;

// Absorption of pure water in m^(-1), wavelength 612.5, 550.0 and 450.0 nanometers
const vec3 a_W = vec3(0.2665,0.0565,0.00922);

// Concentration of chlorophyll, in mg/m^3
const float C_C = 1.0;

// Absorption of chlorophyll, the same wavelengths as above
const vec3 a_C = 0.06 * vec3(0.260, 0.357, 0.944) * pow(C_C, 0.65);

// Concentration of Gelbstoff, in mg/m^3
const float C_Y = 1.0;

// Absorption of Gelbstoff (Fulvic acid and Humic acid), with H / (F + H) = 0.1
const vec3 a_Y = 5.6797 * C_Y * exp(-0.01262 * vec3(612.5, 550.0, 450.0));

const vec3 absorption_coeff = a_W + a_C + a_Y;

// Pure watter scatter coefficients of wavelengths 612.5, 550.0 and 450.0 nanometers in m^(-1)
const vec3 scatter_coeff = vec3(0.000914135, 0.0014584, 0.00348424);

const float GAMMA = 2.2;
const float INVERSE_GAMMA = 1.0 / GAMMA;

vec3 color2light(vec3 color)
{
    return pow(color, vec3(GAMMA));
}

vec3 light2color(vec3 light)
{
    return pow(light, vec3(INVERSE_GAMMA));
}

float power3(float x)
{
    return x * x * x;
}
float power2(float x)
{
    return x * x;
}

vec3 frag2eye(const vec3 normalized_fragCoord)
{
    const vec4 projCoord = vec4(normalized_fragCoord * 2.0 - 1.0, 1.0);
    const vec4 eyeCoord = projInverse * projCoord;
    return vec3(eyeCoord.xyz / eyeCoord.w);
}

vec2 nearby_offset(const vec3 effective_normal, const float strength)
{
    const float effective_strength = strength;
    const vec4 offset_coord = frag_clipCoord + mat2x4(east_clipCoord, north_clipCoord) * (effective_normal.xy * effective_strength);
    return offset_coord.xy / offset_coord.w * 0.5 + 0.5;
}

vec2 flipX( vec2 coord )
{
  return vec2( 1.0 - coord.x, coord.y );
}

float sun_brightness(const vec3 normal, const vec3 eye_dir)
{
    const float normal_fade = smoothstep(500.0, 2500.0, g_LocalEye.z);
    const vec3 effective_normal = normalize(mix(normal, vec3(0.0, 0.0, 1.0), normal_fade));
    
    const vec3 reflect_dir = reflect(-eye_dir, effective_normal);
    const float sun_cos = max(dot(g_SunDir, reflect_dir), 0.0);
    
    const float SUN_RADIUS = 32.0 / 60.0 / 90.0;
    const float brightness_fade = smoothstep(-SUN_RADIUS, SUN_RADIUS, g_SunDir.z);
    
    const float shineness_fade = smoothstep(500.0, 1500.0, g_LocalEye.z);
    const float shineness_low = 800.0, shineness_high = 3200.0;
    const float shineness = mix(shineness_low, shineness_high, shineness_fade);
    
    const float sun_fade = brightness_fade * smoothstep(2000.0, 500.0, g_LocalEye.z);
    //const float sun_fade = brightness_fade;
    return pow(sun_cos, shineness) * sun_fade;
}

float fastFresnel(const vec3 V, const vec3 H, float F0){
	float cosVH = clamp(1-dot(V,H), 0, 1);
	float cosVH5 = cosVH*cosVH*cosVH*cosVH*cosVH;
	return F0 + (1-F0)*cosVH5;
}
void main()
{
#ifndef OCEAN_WIREFRAME
    // Calculate eye vector.
    const vec3 eye_vec = g_LocalEye - local_pos;
    const vec3 eye_dir = normalize(eye_vec);

    // --------------- Blend perlin noise for reducing the tiling artifacts

    // Blend displacement to avoid tiling artifact
    const float dist_2d = length(eye_vec.xy);
    const float blend_factor_raw = (PATCH_BLEND_END - dist_2d) / (PATCH_BLEND_END - PATCH_BLEND_BEGIN);
    const float blend_factor_3d = clamp((PATCH_BLEND_END - length(eye_vec)) / (PATCH_BLEND_END - PATCH_BLEND_BEGIN), 0, 1);
    const float blend_factor = power3(clamp(blend_factor_raw, 0, 1));

    // Compose perlin waves from three octaves
    const vec2 perlin_tc = oceanTexCoord * g_PerlinSize*0.06 + g_UVBase;
    const vec2 perlin_tc0 = perlin_tc * g_PerlinOctave.x + g_PerlinMovement;
    const vec2 perlin_tc1 = perlin_tc * g_PerlinOctave.y + g_PerlinMovement;
    const vec2 perlin_tc2 = perlin_tc * g_PerlinOctave.z + g_PerlinMovement;

    const vec4 perlin_0 = texture(g_texPerlin, perlin_tc0);
    const vec4 perlin_1 = texture(g_texPerlin, perlin_tc1);
    const vec4 perlin_2 = texture(g_texPerlin, perlin_tc2);
    
    const vec4 perlin = (perlin_0 * g_PerlinGradient.x + perlin_1 * g_PerlinGradient.y + perlin_2 * g_PerlinGradient.z);


    // --------------- Water body color

    // Texcoord mash optimization: Texcoord of FFT wave is not required when blend_factor > 1
    const vec2 fft_tc = oceanTexCoord;

    // Gradients in textures are in cm/cm.
	const vec4 gradTapFFT = texture(g_texGradient, fft_tc);
	//const vec4 gradTapFFT = vec4(0);
    //const vec2 gradFFT = gradTapFFT.xy;
	const vec2 gradFFT = -gradTapFFT.yx;
    const vec2 grad = mix(perlin.xy, gradFFT, blend_factor);
	const float foldFFT = clamp(gradTapFFT.w * 0.15, 0, 1);
	//const float foldFFT = clamp((3 - gradTapFFT.w)*clamp(gradTapFFT.x + gradTapFFT.y,0,1)*0.2, 0, 1);
    // Calculate normal here.
	float TexelLength_x2 = 100 / 2048 * 0.2;
    const vec3 normal_raw = normalize(vec3(grad.xy, 6));//TexelLength_x2 has problem
    const vec3 normal = normalize(mix(normal_raw, vec3(0.0, 0.0, 1.0), smoothstep(100f, 1500.0, length(eye_vec))));
    //2019.04.01 Editing
    const vec3 reflect_dir_real = reflect(-eye_dir, normal);
    const float cos_angle = dot(normal, eye_dir);
	//const float fastF = fastFresnel(eye_dir, (normal+eye_dir)/2, 0.02); 
	//const float fastF = fastFresnel(eye_dir, normal, 0.02); 
    // ramp.x for fresnel term. ramp.y for sky blending
    const vec4 ramp_raw = texture(g_texFresnel, cos_angle).xyzw;
    
    // A workaround to deal with "indirect reflection vectors" (which are rays requiring multiple
    // reflections to reach the sky).
    // Nvidia code ignored the situation when reflect_dir_real.z < g_BendParam.y (mix factor > 1.0),
    // also the if-statement is not necessary. They can be solved simply by a clamp.
    // Still I don't know the logic behind this magic bending...
    const vec4 ramp = mix(ramp_raw.zzzz, g_BendParam.zzzz, clamp((g_BendParam.x - reflect_dir_real.z)/(g_BendParam.x - g_BendParam.y), 0.0, 1.0)); //editing
   
    const vec3 reflect_dir = vec3(reflect_dir_real.xy, max(0, reflect_dir_real.z));

    const vec3 normalized_fragCoord = vec3(gl_FragCoord.xy / screenSize, gl_FragCoord.z);
    const vec3 surface_eyePos = frag2eye(normalized_fragCoord);
    
    const float reflect_direct_depth = texture(g_texReflectDepthRawFlip, flipX(normalized_fragCoord.xy)).x;
    const float refract_direct_depth = texture(g_texRefractDepthRaw, normalized_fragCoord.xy).x;
    
    const vec3 reflect_direct_eyePos = frag2eye(vec3(normalized_fragCoord.xy, reflect_direct_depth));
    const vec3 refract_direct_eyePos = frag2eye(vec3(normalized_fragCoord.xy, refract_direct_depth));
    
    const float reflect_direct_length = distance(surface_eyePos, reflect_direct_eyePos);
    const float refract_direct_length = distance(surface_eyePos, refract_direct_eyePos);
    
    const float reflect_strength = clamp(reflect_direct_length * 0.5, 0.0, 10.0);
    const float refract_strength = clamp(refract_direct_length * 0.5, 0.0, 10.0);
    
    const vec2 reflect_texCoord = nearby_offset( normal, reflect_strength);
    const vec2 refract_texCoord = nearby_offset(-normal, refract_strength * 2.0);
    float cos_sun = max(dot(g_SunDir, vec3(0.0, 0.0, 1.0)), 0.001) * 1.2;
	//const vec4 nearby_diffuse = texture(g_texReflectColorFlip, flipX(reflect_texCoord));
    //const vec3 reflect_nearby_light = color2light(nearby_diffuse.rgb) * cos_sun;
    const vec3 refract_nearby_light = texture(g_texRefractColor, refract_texCoord).rgb;
    
    const float reflect_offset_depth = texture(g_texReflectDepthRawFlip, flipX(reflect_texCoord)).x;
    const float refract_offset_depth = texture(g_texRefractDepthRaw, refract_texCoord).x;
    
    const vec3 reflect_offset_eyePos = frag2eye(vec3(reflect_texCoord, reflect_offset_depth));
    const vec3 refract_offset_eyePos = frag2eye(vec3(refract_texCoord, refract_offset_depth));
    
    const float reflect_offset_length = distance(surface_eyePos, reflect_offset_eyePos);
    const float refract_offset_length = distance(surface_eyePos, refract_offset_eyePos);
    
    const vec3 reflect_remote_light_cube = color2light(texture(g_texReflectCube, reflect_dir).rgb);
    const vec3 reflect_remote_light_sky_color = color2light(g_SkyColor);

	//const vec3 reflect_remote_light = mix(reflect_remote_light_sky, reflect_remote_light_cube, 1); // 将ramp.y的值修改为一，不然实际上做了两次菲涅尔反射
    const vec3 refract_remote_light = texture(g_texRefractColor, normalized_fragCoord.xy).rgb;
    
    //const float reflect_remote_factor = texture(g_texReflectDepthNearby, vec3(reflect_texCoord, gl_FragCoord.z)).r + texture(g_texReflectDepthRemote, vec3(reflect_texCoord, 1.0)).r;
    //const float refract_remote_factor = texture(g_texRefractDepthNearby, vec3(refract_texCoord, gl_FragCoord.z)).r;
    
    const vec3 refract_ray_direction = refract_offset_eyePos - surface_eyePos;
    const float refract_ray_offset = dot(normalize(refract_ray_direction), normalize(surface_eyePos));
    
    const float reflect_remote_factor = float(gl_FragCoord.z > reflect_offset_depth);// || (reflect_offset_depth == 1.0));
    //const float refract_remote_factor = float(gl_FragCoord.z > refract_offset_depth);
    const float refract_remote_factor = float(gl_FragCoord.z > refract_offset_depth || refract_ray_offset < 0.86602540378443864676372317075294);
	float near_factor = smoothstep(20.0, 6.0, pow(length(reflect_offset_eyePos), 0.3));
//    const vec3 reflect_light_src = mix(reflect_nearby_light, reflect_remote_light, 
//#ifdef NO_REFLECTION
//	1.0
//#else
//	1.0- nearby_diffuse.a * near_factor
//#endif
//	);
	//const vec3 reflect_light_src = mix(reflect_remote_light_sky, reflect_remote_light_cube, 1);
	const vec3 reflect_light_src = reflect_remote_light_cube*reflect_remote_light_sky_color;
    const vec3 refract_light_src = mix(refract_nearby_light, refract_remote_light, refract_remote_factor);
	//近海岸的黄色
    float dist2polution = 10000000000.0;
	for(int i=0; i<PolutionNum; i++)
		dist2polution = min(dist2polution, length(polution_vec[i]));
	const float polution_factor = pow(smoothstep(50000.0, 500.0, dist2polution),3);
	const vec3 polution_color = vec3(0.5, 0.45, 0.35);
	const vec3 diffuse_color = mix(vec3(1), polution_color, polution_factor);
    if(gl_FrontFacing)
    {
        const float refract_remote_length = refract_direct_length;
        const float refract_mixed_length = mix(refract_offset_length, refract_remote_length, refract_remote_factor);
		const vec3 reflect_light = reflect_light_src;
        //const vec3 refract_light = water_propagate2(absorption_coeff, scatter_coeff, absorption_coeff, refract_light_src, vec3(1.0), refract_mixed_length, -eye_dir.z);
		const vec3 refract_light = water_propagate(absorption_coeff, scatter_coeff, diffuse_color, refract_light_src, refract_mixed_length * (1.0 + eye_dir.z));
        const vec3 sun_light = color2light(g_SunColor) * sun_brightness(normal_raw, eye_dir) ;
		//const vec3 environment_light = color2light(diffuse_color)*0.03f;
        const vec3 foam_light = color2light((foldFFT.xxx) * power3(blend_factor_3d)) ;
		//const vec3 total_light = mix(refract_light, reflect_light, ramp_raw.x) + sun_light + foam_light + environment_light;
		const vec3 total_light = mix(mix(refract_light, reflect_light, ramp_raw.x), color2light(polution_color), polution_factor) + sun_light + foam_light; //修改原本污染区域的颜色混合方式

		color_out.rgb = total_light;
		const float foamRange = pow(smoothstep(0.2f*(1 + g_fft_scale / 12), 2f*(1 + g_fft_scale / 12), gradTapFFT.w), 1)+(oceanBoatGrad.x+ oceanBoatGrad.y)*10;
		const float foamTex = power2(texture(g_texFoam, perlin_tc0 * 2).r) * 2f;
		const float foamAlbedo = clamp(foamTex, 0, 1) * foamRange;

		const float bankFoamWaveDistance = pow(1 - refract_direct_length, 2.2)*smoothstep(0f, 1f, gradTapFFT.y + gradTapFFT.x);
		const float bankFoamWaveColor = power3(texture(g_texFoam, perlin_tc0).r) * 10;
		const vec3 bankFoamWave = vec3(clamp(bankFoamWaveDistance*bankFoamWaveColor, 0, 1));

		const float bankFoamDistance = pow(1 - refract_direct_length*2, 2.2);
		const float bankFoamColor = power3(texture(g_texFoam, perlin_tc0*0.5*g_PerlinSize*0.1f).r);
		const vec3 bankFoam = vec3(clamp(bankFoamDistance*bankFoamColor, 0, 1));

		const float bankFoamDistance2 = pow(1 - refract_direct_length*2, 2.2);
		const float bankFoamColor2 = power3(texture(g_texFoam, perlin_tc0*g_PerlinSize*0.1f).r);
		const vec3 bankFoam2 = vec3(clamp(bankFoamDistance2*bankFoamColor2, 0, 1));

		color_out.rgb = total_light+ vec3(foamAlbedo) * power3(blend_factor_3d) + bankFoamWave * power3(blend_factor_3d) + bankFoam + bankFoam2;
		//color_out.rgb = vec3(oceanBoatGrad.xy,0);
		//color_out.rgb = vec3(boat1_offset,0);
		//color_out.rgb = displacementValue;
		//color_out.rg = vec2(pos2local(lonlat2world(boat1, 1.0), east, north, surface_up));
		//color_out.rgb = vec3(1)-refract_direct_length.xxx/0.5;
		//emit_out = (vec3(foamAlbedo) * power3(blend_factor_3d) + bankFoamWave * blend_factor_3d + bankFoam + bankFoam2)*50;
		//color_out.rgb = refract_light.rgb;
		//color_out.rgb = vec3(0);
		//color_out.rgb = vec3((2-gradTapFFT.w)*clamp(gradTapFFT.x+ gradTapFFT.y,0,1))/5;
		//color_out.rgb = reflect_remote_light.xyz;
		//color_out.rgb = gradTapFFT.xyz;
		//color_out.rgb = (vec3(1)-vec3((1.0f + gradTapFFT.x)*(1.0f + gradTapFFT.z) - gradTapFFT.y*gradTapFFT.w))/3;
		//color_out.rgb = vec3(fft_tc - floor(fft_tc), 0);
		//color_out.rgb = vec3(perlin_tc0 - floor(perlin_tc0), 0);
		//color_out.rgb = vec3(oceanTexCoord - floor(oceanTexCoord), 0);

    }
    else
    {
        //const vec3 refract_light = water_propagate(absorption_coeff, scatter_coeff, vec3(0.3), refract_light_src, length(surface_eyePos));
        color_out.rgb = refract_light_src;
    }
    const mat3 tbn = mat3(local2eye);
    color_out.a = 1.0;
    normal_out = vec4(tbn * normal, 1);
	emit_out = vec3(0,0,0);
    material_out = 4;
#else
    color_out = vec4(1.0);
    normal_out = vec4(0, 0, 0, 1);
    emit_out = vec3(0, 0, 0);
    material_out = 4;
#endif
}

#endif
