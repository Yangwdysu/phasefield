#include "gl_utilities.h"
#include <iostream>
using std::cout;
using std::endl;

namespace gl_utility{

void createTexture(unsigned width, unsigned height, GLint format, GLuint &texture_id, GLint wrapMethod, GLint minFilter, GLint magFilter, GLuint outformat, GLuint  type, GLint anisotropy)
{
	glGetError();
	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, outformat, type, 0);
	int tmp = glGetError();
	if (tmp)
		cout << "error opengl" << tmp << endl;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapMethod);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapMethod);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, wrapMethod);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
	if (anisotropy) glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void initializeTexture(unsigned width, unsigned height, GLint format, GLuint texture_id, GLint wrapMethod, GLint minFilter, GLint magFilter, GLuint outformat, GLuint  type, GLint anisotropy)
{
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, outformat, type, 0);
	int tmp = glGetError();
	if (tmp)
		cout << "error opengl" << tmp << endl;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapMethod);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapMethod);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, wrapMethod);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
	if (anisotropy) glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void createTexture2DArray(GLuint &texture_id, unsigned int width, unsigned int height, unsigned int layer, GLint format, GLint wrapMethod, GLint Filter) {
	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D_ARRAY, texture_id);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, wrapMethod);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, wrapMethod);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, wrapMethod);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, Filter);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, Filter);
	glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, format, width, height, layer);
	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
}

}