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

namespace dnn
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
    inline LogLevel getLogLevel() const
    {
        return logLevel;
    }

    void log(LogLevel level, const std::string& msg) const;

    inline void all(const std::string& msg) const
    {
        log(ALL, msg);
    }

    inline void debug(const std::string& msg) const
    {
        log(DEBUG, msg);
    }

    inline void info(const std::string& msg) const
    {
        log(INFO, msg);
    }

    inline void warning(const std::string& msg) const
    {
        log(WARNING, msg);
    }

    inline void error(const std::string& msg) const
    {
        log(ERROR, msg);
    }

private:
    std::string logName;
    LogLevel logLevel;
};

} // namepace dnn

#endif // DNN_BASE_LOGGER_H
