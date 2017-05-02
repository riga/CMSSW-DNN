/*
 * Logger.
 *
 * Author:
 *   Marcel Rieger
 */

#include "DNN/Base/interface/Logger.h"

namespace dnn
{

Logger::Logger(const std::string& name, LogLevel level)
    : logName(name)
    , logLevel(level)
{
}

Logger::~Logger()
{
}

std::vector<std::string> Logger::getLogLevelNames()
{
    static std::vector<std::string> names = { "ALL", "DEBUG", "INFO", "WARNING", "ERROR" };
    return names;
}

std::string Logger::getLogLevelName(LogLevel level)
{
    return getLogLevelNames()[level];
}

void Logger::setLogLevel(LogLevel level)
{
    log(DEBUG, "set log level to " + std::to_string(level));
    logLevel = level;
}

void Logger::log(LogLevel level, const std::string& msg) const
{
    if (level >= logLevel)
    {
        if (level <= INFO)
        {
            std::cout << logName << ": " << getLogLevelName(level) << ": " << msg << std::endl;
        }
        else
        {
            std::cerr << logName << ": " << getLogLevelName(level) << ": " << msg << std::endl;
        }
    }
}

} // namespace dnn
