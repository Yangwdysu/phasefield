#include <Windows.h>

#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <fstream>
//#include "RigidBody/RigidBody.h"
//#include "CapillaryWave.h"
//#include"PhaseField.h"
using namespace std;
//#include <sys/time.h>

#if __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif





//参数指定正方形位置和大小
GLfloat x1 = 100.0f;
//GLfloat y1 = 150.0f;
GLsizei rsize = 50;

GLfloat y = 150.0f;

//正方形运动变换的步长
GLfloat xstep = 1.0f;
GLfloat ystep = 1.0f;


//窗口大小
GLfloat windowWidth;
GLfloat windowHeight;


//渲染场景
void RenderScene(void)
{
	//用当前清除色清除颜色缓冲区，设定窗口背景色
	glClear(GL_COLOR_BUFFER_BIT);

	//设置当前绘图使用RGB颜色
	glColor3f(0.0f, 1.0f, 0.0f);

	//使用当前颜色绘制一个正方形
	glRectf(x1, y,x1+rsize, y +rsize);

	//清空命令缓冲区并交换帧缓存
	glutSwapBuffers();


}
//改变大小
void ChangeSize(GLsizei w, GLsizei h)
{
	if (h == 0)
		h = 1;
	//设置视区大小
	glViewport(0, 0, w, h);
	//重置坐标系统，使投影变换位置
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//建立修剪空间的范围 修改空间坐标值
	if (w <= h)
	{
		windowHeight = 250.0f*h / w;
		windowWidth = 250.0f;
	}
	else
	{
		windowWidth = 250.0f*w / h;
		windowHeight = 250.0f;
	}
	glOrtho(0.0f, windowWidth, 0.0f, windowHeight, 1.0f, -1.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


}


//时间函数
void TimeFunction(int value)
{
	//处理到达窗口边界的正方形，使它能够进行反弹
	if (x1 >windowWidth - rsize || x1 < 0)
		xstep = -xstep;
	if (y > windowHeight - rsize || y < 0)
		ystep = -ystep;

	if (x1 > windowWidth - rsize)
		x1 = windowWidth - rsize - 1;
	if (y > windowHeight - rsize)
		y = windowHeight - rsize - 1;


	//根据步长修改正方形的位置
	x1 += xstep;
	y += ystep;
	//重新绘图
	glutPostRedisplay();
	glutTimerFunc(33, TimeFunction, 1);


}




//渲染状态
void SetupRC(void)
{


	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}
int main(int argc, char* argv[])
{
	//初始化GLUT库  OpenGL窗口的显示方式
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);//双缓冲窗口|RGB颜色模式
												//创建窗口的标题名称
	glutCreateWindow("OpenGL绘制移动矩形");
	//设置当前窗口显示回调函数和窗口在整形回调函数
	glutDisplayFunc(RenderScene);
	glutReshapeFunc(ChangeSize);
	glutTimerFunc(33, TimeFunction, 1);
	//设置渲染状态
	SetupRC();
	//启动主要的GLUT事件处理循环
	glutMainLoop();
	return 0;
}
