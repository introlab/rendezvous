#include <display/DisplayManager.h>

#include <glad/glad.h>

#ifdef _WIN32
#include <glad/glad_wgl.h>
#endif

#ifdef __linux__
#include <glad/glad_glx.h>
#endif

#include <GLFW/glfw3.h>
#include <iostream>

using namespace std;

GLFWwindow* DisplayManager::m_window = nullptr;
InputCallback* DisplayManager::m_inputCallback = nullptr;

int DisplayManager::createDisplay(int width, int height)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    m_window = glfwCreateWindow(width, height, "Dewarping", NULL, NULL);
    if (m_window == NULL)
    {
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, DisplayManager::framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

#ifdef _WIN32
    if (!gladLoadWGLLoader((GLADloadproc)glfwGetProcAddress, GetWindowDC(0)))
    {
        std::cout << "Failed to initialize WGLAD" << std::endl;
        return -1;
    }

    if (glfwExtensionSupported("WGL_EXT_swap_control"))
    {
        wglSwapIntervalEXT(0);
        std::cout << "Video card supports WGL_EXT_swap_control." << std::endl;
    }
#endif

#ifdef __linux__

	if (!gladLoadGLXLoader((GLADloadproc)glfwGetProcAddress, nullptr, 0))
	{
		std::cout << "Failed to initialize WGLAD" << std::endl;
		return -1;
	}

	if (glfwExtensionSupported("GLX_EXT_swap_control"))
	{
		Display *dpy = glXGetCurrentDisplay();
    	GLXDrawable drawable = glXGetCurrentDrawable();

		if (drawable)
		{
			glXSwapIntervalEXT(dpy, drawable, 0);
		}
		else
		{
			std::cout << "Error when disabling glXSwapIntervalEXT" << std::endl;
		}
		
		std::cout << "Video card supports WGL_EXT_swap_control." << std::endl;
	}

#endif

    return 0;
}

void DisplayManager::updateDisplay()
{
    glfwSwapBuffers(m_window);
    glfwPollEvents();
}

void DisplayManager::closeDisplay()
{
    glfwTerminate();
}

bool DisplayManager::isCloseRequested()
{
    return glfwWindowShouldClose(m_window);
}

void DisplayManager::processInput()
{
    if (m_inputCallback)
    {
        m_inputCallback(m_window);
    }
}

void DisplayManager::registerInputCallback(InputCallback* callback)
{
    m_inputCallback = callback;
}

void DisplayManager::resizeWindow(int width, int height)
{
    glfwSetWindowSize(m_window, width, height);
}

void DisplayManager::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}
