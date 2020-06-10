#pragma once
#include <GL/glew.h>

namespace gl_utility{

void createTexture(unsigned width, unsigned height, GLint format, GLuint &texture_id, GLint wrapMethod, GLint minFilter, GLint magFilter, GLuint outformat, GLuint type, GLint anisotropy = 0);

void initializeTexture(unsigned width, unsigned height, GLint format, GLuint texture_id, GLint wrapMethod, GLint minFilter, GLint magFilter, GLuint outformat, GLuint type, GLint anisotropy = 0);

void createTexture2DArray(GLuint &texture_id, unsigned int width, unsigned int height, unsigned int layer, GLint format, GLint wrapMethod, GLint Filter);

}