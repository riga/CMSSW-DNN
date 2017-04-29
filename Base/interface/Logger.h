/*
 * Logger.
 *
 * Author:
 *   Marcel Rieger
 */

#ifndef DNN_BASE_LOGGER_H
#define DNN_BASE_LOGGER_H

#include <iostream>
#include <string>
#include <vector>

#include "Python.h"

namespace DNN
{

enum LogLevel
{
    ALL = 0,
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger
{
public:
    Logger(const std::string& name, LogLevel level = INFO);
    virtual ~Logger();

    static std::vector<std::string> getLogLevelNames();
    static std::string getLogLevelName(LogLevel level);

    void setLogLevel(LogLevel level);
    LogLevel getLogLevel() const;

    void log(LogLevel level, const std::string& msg) const;
    void all(const std::string& msg) const;
    void debug(const std::string& msg) const;
    void info(const std::string& msg) const;
    void warning(const std::string& msg) const;
    void error(const std::string& msg) const;

private:
    std::string logName;
    LogLevel logLevel;
};

} // namepace DNN

#endif // DNN_BASE_LOGGER_H
