dvec3 lonlat2world(dvec2 lonlat, double radius)
{
    const double sint = sind(lonlat.x);
    const double cost = cosd(lonlat.x);
    const double sinp = sind(lonlat.y);
    const double cosp = cosd(lonlat.y);
    return dvec3(cosp * cost, cosp * sint, sinp) * radius;
}

// Local position is calculated in a unit sphere.
dvec2 pos2local(dvec3 p, dvec3 e, dvec3 n, dvec3 r)
{
    const dvec3 bb = cross(r, p);
    // Use linear approximation for points within one degree
    // Do not work for points on the opposite side
    if(length(bb) < 0.0175 && dot(r, p) > 0)
    {
        const dvec3 worldDist = p - r;
        return dvec2(dot(worldDist, e), dot(worldDist, n));
    }
    else
    {
        const dvec3 b = normalize(bb);
        const dvec3 t = cross(b, r);
        const double sint = dot(cross(n, b), r);
        const double cost = dot(n, b);
        const double cosp = dot(r, p);
        const double phi = acos(float(cosp));
        return dvec2(cost * phi, sint * phi);
    }
}

// local_pos : meter -> (texCoord, offset : meter)
vec3 calcOceanCoords(vec3 pos_local, out vec2 uv_local, out vec2 boat_grad)
{
	// boat trail  use radian, not degree
	dvec3 boat1_pos = lonlat2world(boat1, 1.0);
	vec2 boat1_local = vec2(pos2local(boat1_pos, east, north, surface_up)* EarthRadius);
	boat1_local = pos_local.xy - boat1_local;
	vec2 boat1_uv = boat1_local / g_capillary_real_length - vec2(boat1_offset);
	vec4 boat1_value = textureLod(g_texBoat1, boat1_uv, 0);
	vec3 displacement_boat1 = vec3(0, 0, boat1_value.y);
	boat_grad = boat1_value.zw;



	//uv_local = pos_local.xy * g_UVScale/10;// +g_UVOffset;
	float fft_scale = 1.0/g_fft_resolution*g_fft_real_length;
	uv_local = pos_local.xy / g_fft_real_length;
	//uv_local = pos_local.xy * 1.0 / 512 * 100;
	//uv_local = pos_local.xy * g_UVScale * 0.01 + g_UVOffset;
    
    const vec3 eye_vec = pos_local.xyz - g_LocalEye;
    
    const float dist_2d = length(eye_vec.xy);
    const float blend_factor = clamp((PATCH_BLEND_END*(1+ g_fft_scale/12f) - dist_2d) / (PATCH_BLEND_END*(1 + g_fft_scale / 12f) - PATCH_BLEND_BEGIN), 0, 1);
    
    const vec2 perlin_tc = uv_local * g_PerlinSize + g_UVBase;
    const float perlin_0 = textureLod(g_texPerlin, perlin_tc * g_PerlinOctave.x + g_PerlinMovement, 0).w;
    const float perlin_1 = textureLod(g_texPerlin, perlin_tc * g_PerlinOctave.y + g_PerlinMovement, 0).w;
    const float perlin_2 = textureLod(g_texPerlin, perlin_tc * g_PerlinOctave.z + g_PerlinMovement, 0).w;
    const float height_perlin = perlin_0 * g_PerlinAmplitude.x + perlin_1 * g_PerlinAmplitude.y + perlin_2 * g_PerlinAmplitude.z;
    
    const vec3 displacement_perlin = vec3(0, 0, height_perlin)*0.01;
    //const vec3 displacement_fft = textureLod(g_texDisplacement, uv_local, 0).xyz;
	vec3 displacement_fft = textureLod(g_texDisplacement, uv_local, 0).xzy*fft_scale;
	displacement_fft.xy = g_chopiness*displacement_fft.xy;
	//const vec3 displacement_fft = texture(g_texDisplacement, uv_local);
    const vec3 displacement = mix(displacement_perlin, displacement_fft, blend_factor);
    
    // Convert cm -> m
	//return displacement_fft*0.2+displacement_perlin*0.01;
	//return displacement_fft;
	//return vec3(boat1_uv-floor(boat1_uv), 0);
	//return vec3(boat1_local, 0);
	//return displacement_fft;
	//return displacement_fft*0.01;
	//return vec3(boat_grad, 0);
	//return displacement_boat1;
	//return displacement + displacement_boat1;
    return displacement*0.3+displacement_boat1*3;
}
