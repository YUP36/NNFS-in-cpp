#ifndef NAMEDWRAPPER_H
#define NAMEDWRAPPER_H

#include <string>

class NamedWrapper {

    public:
        NamedWrapper();
        virtual std::string getName() const;

};

#endif