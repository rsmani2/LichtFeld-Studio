/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "subprocess.hpp"

#include <core/logger.hpp>

#ifndef _WIN32
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace lfs::python {

    SubProcess::~SubProcess() {
        kill();
    }

#ifndef _WIN32

    bool SubProcess::start(const std::string& program, const std::vector<std::string>& args) {
        kill();

        int pipefd[2];
        if (pipe(pipefd) == -1) {
            LOG_ERROR("pipe() failed: {}", strerror(errno));
            return false;
        }

        pid_ = fork();
        if (pid_ == -1) {
            LOG_ERROR("fork() failed: {}", strerror(errno));
            close(pipefd[0]);
            close(pipefd[1]);
            return false;
        }

        if (pid_ == 0) {
            // Child process
            close(pipefd[0]);

            dup2(pipefd[1], STDOUT_FILENO);
            dup2(pipefd[1], STDERR_FILENO);
            close(pipefd[1]);

            std::vector<const char*> argv;
            argv.push_back(program.c_str());
            for (const auto& arg : args) {
                argv.push_back(arg.c_str());
            }
            argv.push_back(nullptr);

            execvp(program.c_str(), const_cast<char* const*>(argv.data()));
            _exit(127);
        }

        // Parent process
        close(pipefd[1]);
        stdout_fd_ = pipefd[0];

        int flags = fcntl(stdout_fd_, F_GETFL, 0);
        fcntl(stdout_fd_, F_SETFL, flags | O_NONBLOCK);

        LOG_INFO("Subprocess started: {} (pid {})", program, pid_);
        return true;
    }

    ssize_t SubProcess::read(char* buf, size_t len) {
        if (stdout_fd_ < 0)
            return -1;

        ssize_t n = ::read(stdout_fd_, buf, len);
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return 0;
        }
        return n;
    }

    bool SubProcess::is_running() const {
        if (pid_ <= 0)
            return false;
        int status;
        pid_t result = waitpid(pid_, &status, WNOHANG);
        if (result == pid_) {
            // Process has exited, capture exit code
            if (WIFEXITED(status)) {
                const_cast<SubProcess*>(this)->exit_code_ = WEXITSTATUS(status);
            } else if (WIFSIGNALED(status)) {
                const_cast<SubProcess*>(this)->exit_code_ = 128 + WTERMSIG(status);
            }
            const_cast<SubProcess*>(this)->pid_ = -1;
            return false;
        }
        return result == 0;
    }

    void SubProcess::kill() {
        if (stdout_fd_ >= 0) {
            close(stdout_fd_);
            stdout_fd_ = -1;
        }
        if (pid_ > 0) {
            ::kill(pid_, SIGTERM);
            usleep(50000);
            if (is_running()) {
                ::kill(pid_, SIGKILL);
            }
            int status;
            waitpid(pid_, &status, 0);
            if (WIFEXITED(status)) {
                exit_code_ = WEXITSTATUS(status);
            }
            pid_ = -1;
        }
    }

    int SubProcess::wait() {
        // If process already reaped by is_running(), return cached exit code
        if (pid_ <= 0)
            return exit_code_;

        // Close pipe before waiting (so process isn't blocked on full pipe)
        if (stdout_fd_ >= 0) {
            close(stdout_fd_);
            stdout_fd_ = -1;
        }

        int status;
        if (waitpid(pid_, &status, 0) == pid_) {
            if (WIFEXITED(status)) {
                exit_code_ = WEXITSTATUS(status);
            } else if (WIFSIGNALED(status)) {
                exit_code_ = 128 + WTERMSIG(status);
            }
        }
        pid_ = -1;
        return exit_code_;
    }

#else // Windows

    bool SubProcess::start(const std::string& program, const std::vector<std::string>& args) {
        kill();

        SECURITY_ATTRIBUTES sa;
        sa.nLength = sizeof(SECURITY_ATTRIBUTES);
        sa.bInheritHandle = TRUE;
        sa.lpSecurityDescriptor = nullptr;

        HANDLE hRead, hWrite;
        if (!CreatePipe(&hRead, &hWrite, &sa, 0)) {
            LOG_ERROR("CreatePipe failed");
            return false;
        }
        SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOW si = {};
        si.cb = sizeof(si);
        si.hStdOutput = hWrite;
        si.hStdError = hWrite;
        si.dwFlags |= STARTF_USESTDHANDLES;

        std::wstring cmdline;
        cmdline += L"\"";
        cmdline += std::wstring(program.begin(), program.end());
        cmdline += L"\"";
        for (const auto& arg : args) {
            cmdline += L" \"";
            cmdline += std::wstring(arg.begin(), arg.end());
            cmdline += L"\"";
        }

        PROCESS_INFORMATION pi = {};
        if (!CreateProcessW(nullptr, cmdline.data(), nullptr, nullptr, TRUE,
                            CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
            CloseHandle(hRead);
            CloseHandle(hWrite);
            LOG_ERROR("CreateProcessW failed");
            return false;
        }

        CloseHandle(hWrite);
        CloseHandle(pi.hThread);
        process_ = pi.hProcess;
        pipe_stdout_ = hRead;

        LOG_INFO("Subprocess started: {}", program);
        return true;
    }

    ssize_t SubProcess::read(char* buf, size_t len) {
        if (pipe_stdout_ == INVALID_HANDLE_VALUE)
            return -1;

        DWORD available = 0;
        if (!PeekNamedPipe(pipe_stdout_, nullptr, 0, nullptr, &available, nullptr) || available == 0) {
            return 0;
        }

        DWORD bytesRead = 0;
        if (!ReadFile(pipe_stdout_, buf, static_cast<DWORD>(len), &bytesRead, nullptr)) {
            return -1;
        }
        return static_cast<ssize_t>(bytesRead);
    }

    bool SubProcess::is_running() const {
        if (process_ == INVALID_HANDLE_VALUE)
            return false;
        DWORD code;
        return GetExitCodeProcess(process_, &code) && code == STILL_ACTIVE;
    }

    void SubProcess::kill() {
        if (pipe_stdout_ != INVALID_HANDLE_VALUE) {
            CloseHandle(pipe_stdout_);
            pipe_stdout_ = INVALID_HANDLE_VALUE;
        }
        if (process_ != INVALID_HANDLE_VALUE) {
            TerminateProcess(process_, 1);
            WaitForSingleObject(process_, 100);
            DWORD code;
            GetExitCodeProcess(process_, &code);
            exit_code_ = static_cast<int>(code);
            CloseHandle(process_);
            process_ = INVALID_HANDLE_VALUE;
        }
    }

    int SubProcess::wait() {
        if (process_ == INVALID_HANDLE_VALUE)
            return exit_code_;

        if (pipe_stdout_ != INVALID_HANDLE_VALUE) {
            CloseHandle(pipe_stdout_);
            pipe_stdout_ = INVALID_HANDLE_VALUE;
        }

        WaitForSingleObject(process_, INFINITE);
        DWORD code;
        GetExitCodeProcess(process_, &code);
        exit_code_ = static_cast<int>(code);
        CloseHandle(process_);
        process_ = INVALID_HANDLE_VALUE;
        return exit_code_;
    }

#endif

} // namespace lfs::python
